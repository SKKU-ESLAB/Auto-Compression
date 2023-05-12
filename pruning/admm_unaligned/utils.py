import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy

def param_check(name, param, args):
    if (name.split('.')[-1] == "weight"
            and param.dim() == 4
            and param.size(2) == 1
            and param.size(3) == 1):
        return True
    else:
        return False


# Assume that W is a 2-dimensional numpy tensor
# Unaligned pattern
#   10010001
#   10110111
#   00100110
def get_unaligned_mask(W, GS=(4, 1), norm_policy='l2', threshold=None, target_M=1):
    R, C = W.shape
    GR, GC = GS
    w = W.reshape(R, C//GC, GC)

    if norm_policy == 'l1':
        w = np.sum(np.abs(w), axis=2)
    elif norm_policy == 'l2':
        w = np.sum(np.power(w, 2), axis=2)
    else:
        raise ValueError('norm_policy must be l1 or l2')

    # cg_w: column grouped w -> len(cg_w) = R * C // GC = N
    cg_w = w.transpose(1, 0).flatten()
    N = len(cg_w)

    # Grouped weight matrix for all range(N)
    g_w = np.sum(
            np.lib.stride_tricks.as_strided(
                cg_w, shape=(N-GR+1, GR), strides=(cg_w.itemsize, cg_w.itemsize)
            ), axis=1)

    boundary_check = np.where((np.arange(len(g_w)) % R) <= (R - GR), 1, 0)
    # If g_w value contain boundary -> zero value
    g_w = g_w * boundary_check

    # Optimal solution (sum of weights, column indices)
    M = N // GR
    T = np.zeros(M+1)
    if norm_policy == 'l1':
        T[1:] = -np.sum(np.abs(W))
    elif norm_ploicy == 'l2':
        T[1:] = -np.sum(np.power(W, 2))

    g = GR
    G = np.zeros((g-1, int(np.ceil(N/(g-1))) + (M-2)))
    for i in range(len(g_w)):
        G[i%(g-1), int(np.floor(i/(g-1)))] = g_w[i]

    # Choice matrix
    target_i = M + (M - target_M) * (g - 1)
    Cm = np.zeros((target_i, target_M))

    for i in range(1, target_i + 1):
        k1 = (i-1) % (g-1)
        k2 = int(np.floor((i-1)/(g-1)))

        adaptive_M = min(i, target_M)

        no_choose = T[1:adaptive_M+1]
        new_choose = T[0:adaptive_M] + G[k1, k2:k2+adaptive_M]

        Cm[i-1, :adaptive_M] = np.where(no_choose < new_choose, 1, 0)

        T[1:adaptive_M+1] = np.maximum(no_choose, new_choose)


    # Reverse traversal
    j = target_M
    real_i = N
    mask = np.zeros(R * C)
    for i in range(target_i, 0, -1):
        if Cm[i-1, j-1] == 1:
            mask[real_i-g:real_i] = 1
            real_i -= g
            j -= 1
        else:
            real_i -= 1

        if j == 0:
            break

    mask = mask.reshape(C, R).transpose(1, 0)

    return mask

# Assume that W is a 2-dimensional numpy tensor
# Unaligned pattern
#   10010001
#   10110111
#   00100110
def calc_unaligned_greedy(W, GS=(4, 1), norm_policy='l2', threshold=None, min_sparsity=None, target_M=None):
    R, C = W.shape
    GR, GC = GS
    w = W.reshape(R, C//GC, GC)

    if norm_policy == 'l1':
        w = np.sum(np.abs(w), axis=2)
    elif norm_policy == 'l2':
        w = np.sum(np.power(w, 2), axis=2)
    else:
        raise ValueError('norm_policy must be l1 or l2')

    # cg_w: column grouped w -> len(cg_w) = R * C // GC = N
    cg_w = w.transpose(1, 0).flatten()
    N = len(cg_w)

    # Grouped weight matrix for all range(N)
    g_w = np.sum(
            np.lib.stride_tricks.as_strided(
                cg_w, shape=(N-GR+1, GR), strides=(cg_w.itemsize, cg_w.itemsize)
            ), axis=1)

    if norm_policy == 'l1':
        small_val = -np.sum(np.abs(W))
    elif norm_policy == 'l2':
        small_val = -np.sum(np.power(W, 2))
    boundary_check = np.where((np.arange(len(g_w)) % R) <= (R - GR), False, True)
    # If g_w value contain boundary -> zero value
    g_w[boundary_check] = small_val

    # Optimal solution (sum of weights, column indices)
    M = N // GR
    g = GR

    if min_sparsity is not None:
        max_M = int(M * (1 - min_sparsity))
    elif target_M is not None:
        max_M = target_M

    for _ in range(g-1):
        g_w = np.append(g_w, small_val)
    g_w = g_w.reshape(C, R)

    idx_list = []
    score_list = np.array([0. for _ in range(M+1)])
    argmax_idx_list = np.argmax(g_w, axis=1)
    amax_list = np.amax(g_w, axis=1)
    for i in range(max_M):
        c = np.argmax(amax_list)
        r = argmax_idx_list[c]
        idx_list.append((c, r))
        score_list[i+1] = score_list[i] + g_w[c, r]

        start_idx = max(0, r-g+1)
        g_w[c, start_idx:r+g] = small_val
        argmax_idx_list[c] = np.argmax(g_w[c])
        amax_list[c] = g_w[c, argmax_idx_list[c]]

    score_list[max_M+1:] = score_list[max_M]

    mask = np.ones((C, R), dtype=bool)
    for c, r in idx_list:
        mask[c, r:r+g] = False
    mask = mask.transpose(1, 0)

    return score_list, mask




def admm_loss(args, criterion, model, Z, U, output, target):
    idx = 0
    loss = criterion(output, target)
    for name, param in model.named_parameters():
        if param_check(name, param):
            u = U[idx].to(param.device)
            z = Z[idx].to(param.device)
            loss += args.rho / 2 * (param - z + u).norm()
            idx += 1
    return loss


def initialize_Z_and_U(model, args):
    Z = ()
    U = ()
    num_nnz_block_list = []
    for name, param in model.named_parameters():
        if param_check(name, param):
            Z += (param.detach().cpu().clone(),)
            U += (torch.zeros_like(param).cpu(),)
            if args.sparsity_method == "gt":
                num_nnz_block_list.append(0)
    args.num_nnz_block_list = num_nnz_block_list

    return Z, U


def update_X(model):
    X = ()
    for name, param in model.named_parameters():
        if param_check(name, param):
            X += (param.detach().cpu().clone(),)
    return X


def update_Z(X, U, args):
    new_Z = ()

    if args.sparsity_method == "gt":
        score_list = []
        for x, u in zip(X, U):
            z = x + u
            co, ci, kh, kw = z.shape
            if args.unaligned and args.unaligned_score:
                z_np = z.numpy().reshape(co, ci)
                #accumulated_score = calc_unaligned_score(z_np, GS=(args.block_size, 1), norm_policy='l2', min_sparsity=args.min_sparsity)
                accumulated_score, _ = calc_unaligned_greedy(z_np, GS=(args.block_size, 1), norm_policy='l2', min_sparsity=args.min_sparsity)
                score = accumulated_score[1:] - accumulated_score[:-1]
                score_list.append(score)
            else:
                m = z.reshape(co // args.block_size, args.block_size, ci, kh, kw).pow(2).sum(1)
                pcen = np.percentile(m, 100*args.min_sparsity)
                m[m < pcen] = 0.
                score_list.append(m.flatten())
        scores = np.concatenate(score_list)
        global_threshold = np.percentile(scores, 100*args.target_sparsity)

        for i, score in enumerate(score_list):
            args.num_nnz_block_list[i] = int(np.sum(np.where(score < global_threshold, 0, 1)))
            print(args.num_nnz_block_list[i], args.num_nnz_block_list[i]/len(score))

    idx = 0
    for x, u in zip(X, U):
        z = x + u
        co, ci, kh, kw = z.shape
        if args.unaligned:
            if args.sparsity_method == "uniform":
                target_num_block = int(co * ci * kh * kw / args.block_size * (1 - args.target_sparsity))
            elif args.sparsity_method == "gt":
                target_num_block = args.num_nnz_block_list[idx]
            z_np = z.numpy().reshape(co, ci)
            #mask = get_unaligned_mask(z_np, GS=(args.block_size, 1), norm_policy='l2', target_M=target_num_block)
            _, mask = calc_unaligned_greedy(z_np, GS=(args.block_size, 1), norm_policy='l2', target_M=target_num_block)
            under_threshold = torch.BoolTensor(mask.reshape(z.shape))
        else:
            m = z.reshape(co // args.block_size, args.block_size, ci, kh, kw).pow(2).sum(1)
            if args.sparsity_method == "uniform":
                target_sparsity = args.target_sparsity
            elif args.sparsity_method == "gt":
                target_sparsity = 1. - args.num_nnz_block_list[idx] / (co * ci * kh * kw / args.block_size)
            pcen = np.percentile(abs(m), 100*target_sparsity)
            under_threshold = abs(m) < pcen
            under_threshold = under_threshold.repeat_interleave(args.block_size, dim=0)
        z.data[under_threshold] = 0
        new_Z += (z,)
        idx += 1

    return new_Z


def update_Z_l1(X, U, args):
    new_Z = ()
    delta = args.alpha / args.rho
    for x, u in zip(X, U):
        z = x + u
        new_z = z.clone()
        if (z > delta).sum() != 0:
            new_z[z > delta] = z[z > delta] - delta
        if (z < -delta).sum() != 0:
            new_z[z < -delta] = z[z < -delta] + delta
        if (abs(z) <= delta).sum() != 0:
            new_z[abs(z) <= delta] = 0
        new_Z += (new_z,)
    return new_Z


def update_U(U, X, Z):
    new_U = ()
    for u, x, z in zip(U, X, Z):
        new_u = u + x - z
        new_U += (new_u,)
    return new_U


def prune_weight(weight, args, idx):
    # to work with admm, we calculate percentile based on all elements instead of nonzero elements.
    weight_numpy = weight.detach().cpu().numpy()
    co, ci, kh, kw = weight.shape
    if args.unaligned:
        if args.sparsity_method == "uniform":
            target_num_block = int(co * ci * kh * kw / args.block_size * (1 - args.target_sparsity))
        elif args.sparsity_method == "gt":
            target_num_block = args.num_nnz_block_list[idx]
        w = weight_numpy.reshape(co, ci)
        #m = get_unaligned_mask(w, GS=(args.block_size, 1), norm_policy='l2', target_M=target_num_block)
        _, m = calc_unaligned_greedy(w, GS=(args.block_size, 1), norm_policy='l2', target_M=target_num_block)
        under_threshold = m.reshape(weight.shape)
    else:
        m = weight.reshape(co // args.block_size, args.block_size, ci, kh, kw).pow(2).sum(1).detach().cpu().numpy()
        if args.sparsity_method == "uniform":
            target_sparsity = args.target_sparsity
        elif args.sparsity_method == "gt":
            target_sparsity = 1. - args.num_nnz_block_list[idx] / (co * ci * kh * kw / args.block_size)
        pcen = np.percentile(abs(m), 100*target_sparsity)
        under_threshold = abs(m) < pcen
        under_threshold = under_threshold.repeat(args.block_size, axis=0)
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(1 - under_threshold).to(weight.device)
    return mask


def prune_l1_weight(weight, device, delta):
    weight_numpy = weight.detach().cpu().numpy()
    under_threshold = abs(weight_numpy) < delta
    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(abs(weight_numpy) >= delta).to(device)
    return mask


def apply_prune(model, args):
    # returns dictionary of non_zero_values' indices
    print("Apply Pruning based on percentile")
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if param_check(name, param):
            mask = prune_weight(param, args, idx)
            param.data.mul_(mask)
            # param.data = torch.Tensor(weight_pruned).to(device)
            dict_mask[name] = mask
            idx += 1
    return dict_mask

def prune_with_mask(model, dict_mask):
    for name, param in model.named_parameters():
        if param_check(name, param):
            mask = dict_mask[name]
            param.data.mul_(mask)

def apply_l1_prune(model, device, args):
    delta = args.alpha / args.rho
    print("Apply Pruning based on percentile")
    dict_mask = {}
    idx = 0
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight":
            mask = prune_l1_weight(param, device, delta)
            param.data.mul_(mask)
            dict_mask[name] = mask
            idx += 1
    return dict_mask


def print_convergence(model, X, Z):
    idx = 0
    print("normalized norm of (weight - projection)")
    for name, param in model.named_parameters():
        if param_check(name, param):
            x, z = X[idx], Z[idx]
            print("({}): {:.4f}".format(name, (x-z).norm().item() / x.norm().item()))
            idx += 1


def print_prune(model):
    prune_param, total_param = 0, 0
    for name, param in model.named_parameters():
        if param_check(name, param):
            print("[at weight {}]".format(name))
            print("percentage of pruned: {:.4f}%".format(100 * (abs(param) == 0).sum().item() / param.numel()))
            print("nonzero parameters after pruning: {} / {}\n".format((param != 0).sum().item(), param.numel()))
        total_param += param.numel()
        prune_param += (param != 0).sum().item()
    print("total nonzero parameters after pruning: {} / {} ({:.4f}%)".
          format(prune_param, total_param,
                 100 * (total_param - prune_param) / total_param))
