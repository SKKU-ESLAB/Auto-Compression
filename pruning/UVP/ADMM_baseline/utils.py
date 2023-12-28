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

# assume that input is absolute matrix
def greedy_search_unaligned_v2(input, GS, target_M, balanced=False):
    I = np.abs(input.transpose(1, 0))
    GS = tuple(reversed(GS))

    R, C = I.shape
    GR, GC = GS

    rg_I = I.reshape(R // GR, GR, C)
    rg_I = np.sum(rg_I, axis=1)
    small_val = 0

    max_nnz_per_col = np.ceil(target_M / (C - GC + 1))
    nnz_per_col = np.zeros(C - GC + 1)
    selected_index_list = [np.array([]).astype("int") for r in range(R)]

    value_list = [rg_I[r] for r in range(R)]
    index_list = [np.arange(C) for r in range(R)]
    block_list = [np.sum(np.lib.stride_tricks.sliding_window_view(value_list[r], GC, axis=0), axis=1) for r in range(R)]

    score_list = np.zeros((target_M + 1,), dtype=np.float32)
    argmax_idx_list = np.array([np.argmax(block_list[r]) for r in range(R)])
    amax_list = np.array([block_list[r][argmax_idx_list[r]] for r in range(R)])

    mask = np.ones((R, C), dtype=bool)
    for i in range(target_M):
        r = np.argmax(amax_list)
        c = argmax_idx_list[r]

        def _get_new_nnz_per_col(r, c):
            tmp_nnz_per_col = copy.copy(nnz_per_col)
            start_index_list = selected_index_list[r].reshape(-1, GC)[:, 0]
            tmp_nnz_per_col[start_index_list] -= 1

            new_selected_index_list = np.sort(np.concatenate([selected_index_list[r], index_list[r][c:c+GC]]))
            new_start_index_list = new_selected_index_list.reshape(-1, GC)[:, 0]
            tmp_nnz_per_col[new_start_index_list] += 1

            return tmp_nnz_per_col

        if balanced:
            new_nnz_per_col = _get_new_nnz_per_col(r, c)
            while max(new_nnz_per_col) > max_nnz_per_col:
                block_list[r][c] = 0
                argmax_idx_list[r] = np.argmax(block_list[r])
                amax_list[r] = block_list[r][argmax_idx_list[r]]
                r = np.argmax(amax_list)
                c = argmax_idx_list[r]

                new_nnz_per_col = _get_new_nnz_per_col(r, c)

            nnz_per_col = new_nnz_per_col
            selected_index_list[r] = np.sort(np.concatenate([selected_index_list[r], index_list[r][c:c+GC]]))

        score_list[i + 1] = score_list[i] + block_list[r][c]

        mask[r*GR:(r+1)*GR, index_list[r][c:c+GC]] = False

        if len(value_list[r]) > GC:
            #value_list[r][c:c+GC]
            value_list[r] = np.concatenate([value_list[r][:c], value_list[r][c+GC:]])
            #index_list[r][c:c+GC]
            index_list[r] = np.concatenate([index_list[r][:c], index_list[r][c+GC:]])
            block_list[r] = np.sum(np.lib.stride_tricks.sliding_window_view(value_list[r], GC, axis=0), axis=1)
            argmax_idx_list[r] = np.argmax(block_list[r])
            amax_list[r] = block_list[r][argmax_idx_list[r]]
        else:
            amax_list[r] = 0

    mask = mask.transpose(1, 0)

    return score_list, mask

def search_aligned(input, GS, target_M, balanced=False):
    R, C = input.shape
    GR, GC = GS
    abs_I = np.abs(input)
    reshaped_I = abs_I.reshape(R//GR, GR, C//GC, GC)
    g_I = np.sum(reshaped_I, axis=(1, 3))

    if balanced:
        nnz_per_row = target_M // (R // GR)
        if target_M > nnz_per_row * (R // GR):
            nnz_per_row += 1
        g_I = np.where(g_I >= np.sort(g_I)[:, -nnz_per_row][:, np.newaxis], g_I, 0)
    mask = g_I < np.sort(g_I.flatten())[-target_M]
    mask = np.repeat(mask, GR, axis=0)
    mask = np.repeat(mask, GC, axis=1)
    score = abs_I[~mask].sum()
    return [score], mask

def cosine_similarity(A, B):
    dp = np.dot(A, B.T)
    p1 = np.sqrt(np.sum(A**2, axis=1))[:, np.newaxis] + 1e-9
    p2 = np.sqrt(np.sum(B**2, axis=1))[np.newaxis, :] + 1e-9
    return dp / (p1 * p2)


def search_perm(weight, mask, vs, args):
    R = weight.shape[0]
    mask = ~mask
    weight = np.abs(weight)
    best_perm = []
    remained_perm = [r for r in range(R)]

    masked_weight = mask * weight
    cs = cosine_similarity(masked_weight, masked_weight)
    #coef = (1 + np.arange(vs - 1)).reshape(-1, 1)
    if args.unaligned:
        if vs > 2:
            coef = 1 - (1-args.cp_alpha) * (np.arange(vs - 1).reshape(-1, 1) / (vs - 2)) ** args.cp_beta
        else:
            coef = (1 + np.arange(vs - 1)).reshape(-1, 1)

        for r in range(R):
            if r == 0:
                best_i = 0
            else:
                l = min(vs - 1, len(best_perm))
                best_i = np.argmax(np.sum(cs[best_perm[::-1][:l]][:, remained_perm] * coef[:l], axis=0))
            best_perm.append(remained_perm.pop(best_i))
    else:
        for r in range(R):
            if r == 0:
                best_i = 0
            elif r % vs == 0:
                best_i = np.argmin(cs[best_perm[-1], remained_perm])
            else:
                l = min(vs - 1, len(best_perm) % vs)
                best_i = np.argmax(np.sum(cs[best_perm[-l:]][:, remained_perm], axis=0))
            best_perm.append(remained_perm.pop(best_i))

    return np.array(best_perm)

def get_admm_loss(args, model, Z, U):
    idx = 0
    loss = 0
    for name, param in model.named_parameters():
        if param_check(name, param, args):
            u = U[idx].to(param.device)
            z = Z[idx].to(param.device)
            loss += (param - z + u).pow(2).sum()
            idx += 1
    loss = loss * args.rho / 2
    return loss

def initialize_perm_list(model, args):
    perm_list = []
    for name, param in model.named_parameters():
        if param_check(name, param, args):
            perm_list.append(np.arange(param.shape[0]))
    return perm_list

def initialize_Z_and_U(model, args):
    Z = ()
    U = ()
    num_nnz_block_list = []
    for name, param in model.named_parameters():
        if param_check(name, param, args):
            Z += (param.detach().cpu().clone(),)
            U += (torch.zeros_like(param).cpu(),)
            if args.sparsity_method == "gt":
                num_nnz_block_list.append(0)
    args.num_nnz_block_list = num_nnz_block_list

    return Z, U


def update_X(model, args):
    X = ()
    for name, param in model.named_parameters():
        if param_check(name, param, args):
            X += (param.detach().cpu().clone(),)
    return X


def update_Z(X, U, args, perm_list, channel_permute):
    new_Z = ()

    if args.sparsity_method == "gt":
        score_list = []
        for x, u in zip(X, U):
            z = x + u
            co, ci, kh, kw = z.shape
            if args.group_norm == 'l1':
                m = z.reshape(co // args.vector_size, args.vector_size, ci, kh, kw).abs().sum(1)
            elif args.group_norm == 'l2':
                m = z.reshape(co // args.vector_size, args.vector_size, ci, kh, kw).pow(2).sum(1)
            score_list.append(m.flatten())
        scores = np.concatenate(score_list)
        global_threshold = np.percentile(scores, 100*args.target_sparsity)

        for i, score in enumerate(score_list):
            args.num_nnz_block_list[i] = int(np.sum(np.where(score < global_threshold, 0, 1)))

    idx = 0
    score_diff_dict = {}
    for x, u in zip(X, U):
        z = x + u
        co, ci, kh, kw = z.shape

        num_zones = 1
        if not args.unaligned:
            num_zones = co // args.vector_size

        if args.sparsity_method == "uniform":
            target_num_block = int(co * ci * kh * kw / args.vector_size * (1 - args.target_sparsity))
        elif args.sparsity_method == "gt":
            target_num_block = args.num_nnz_block_list[idx]
        z_np = z.numpy().reshape(co, ci)

        original_z_np = z_np

        if args.cp:
            # get original mask without new channel permutation
            if args.unaligned:
                #score_list, permed_mask = calc_unaligned_greedy(z_np[perm_list[idx]], GS=(args.vector_size, 1), norm_policy=args.group_norm, target_M=target_num_block)
                score_list, permed_mask = greedy_search_unaligned_v2(z_np[perm_list[idx]], GS=(args.vector_size, 1), target_M=target_num_block)
            else:
                score_list, permed_mask = search_aligned(z_np[perm_list[idx]], GS=(args.vector_size, 1), target_M=target_num_block)
            mask = permed_mask[np.argsort(perm_list[idx])]
            score = score_list[-1]

            if channel_permute:
                # get element-level mask and do channel permutation search algorithm
                _, element_level_mask = search_aligned(z_np, GS=(1, 1), target_M=target_num_block*args.vector_size)
                cp_perm = search_perm(original_z_np, element_level_mask, args.vector_size, args)

                # get cp_mask and cp_score
                if args.unaligned:
                    #score_list, permed_mask = calc_unaligned_greedy(z_np[cp_perm], GS=(args.vector_size, 1), norm_policy=args.group_norm, target_M=target_num_block)
                    score_list, permed_mask = greedy_search_unaligned_v2(z_np[cp_perm], GS=(args.vector_size, 1), target_M=target_num_block)
                else:
                    score_list, permed_mask = search_aligned(z_np[cp_perm], GS=(args.vector_size, 1), target_M=target_num_block)
                cp_mask = permed_mask[np.argsort(cp_perm)]
                cp_score = score_list[-1]

                score_diff_dict[f"score_diff_L{idx}"] = cp_score - score

                mask = cp_mask
                perm_list[idx] = cp_perm
        else:
            if args.unaligned:
                #_, mask = calc_unaligned_greedy(z_np, GS=(args.vector_size, 1), norm_policy=args.group_norm, target_M=target_num_block)
                _, mask = greedy_search_unaligned_v2(z_np, GS=(args.vector_size, 1), target_M=target_num_block)
            else:
                _, mask = search_aligned(z_np, GS=(args.vector_size, 1), target_M=target_num_block)
        under_threshold = torch.BoolTensor(mask.reshape(z.shape))

        z.data[under_threshold] = 0
        new_Z += (z,)
        idx += 1

    return new_Z, score_diff_dict


def update_U(U, X, Z):
    new_U = ()
    for u, x, z in zip(U, X, Z):
        new_u = u + x - z
        new_U += (new_u,)
    return new_U


def prune_weight(weight, args, idx, perm):
    # to work with admm, we calculate percentile based on all elements instead of nonzero elements.
    weight_numpy = weight.detach().cpu().numpy()
    co, ci, kh, kw = weight.shape

    if args.sparsity_method == "uniform":
        target_num_block = int(co * ci * kh * kw / args.vector_size * (1 - args.target_sparsity))
    elif args.sparsity_method == "gt":
        target_num_block = args.num_nnz_block_list[idx]
    w = weight_numpy.reshape(co, ci)

    if args.cp_ft:
        _, element_level_mask = search_aligned(w, GS=(1, 1), target_M=target_num_block*args.vector_size)
        perm = search_perm(w, element_level_mask, args.vector_size, args)

    if args.cp:
        if args.unaligned:
            #_, permed_m = calc_unaligned_greedy(w[perm], GS=(args.vector_size, 1), norm_policy=args.group_norm, target_M=target_num_block)
            _, permed_m = greedy_search_unaligned_v2(w[perm], GS=(args.vector_size, 1), target_M=target_num_block)
        else:
            _, permed_m = search_aligned(w[perm], GS=(args.vector_size, 1), target_M=target_num_block)
        m = permed_m[np.argsort(perm)]
    else:
        if args.unaligned:
            #_, m = calc_unaligned_greedy(w, GS=(args.vector_size, 1), norm_policy=args.group_norm, target_M=target_num_block)
            _, m = greedy_search_unaligned_v2(w, GS=(args.vector_size, 1), target_M=target_num_block)
        else:
            _, m = search_aligned(w, GS=(args.vector_size, 1), target_M=target_num_block)
    under_threshold = m.reshape(weight.shape)

    weight_numpy[under_threshold] = 0
    mask = torch.Tensor(1 - under_threshold).to(weight.device)
    return mask


def apply_masked_weight(mod, input):
    mod.weight.data.mul_(mod.mask)


def apply_prune(model, args, perm_list):
    # returns dictionary of non_zero_values' indices
    print("Apply Pruning based on percentile")
    idx = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            param = module.weight
            if param_check('.weight', param, args):
                mask = prune_weight(param, args, idx, perm_list[idx])
                param.data.mul_(mask)
                module.register_buffer("mask", mask)
                module.register_forward_pre_hook(
                    lambda mod, input: apply_masked_weight(mod, input))
                idx += 1


def print_prune(model, args):
    prune_param, total_param = 0, 0
    for name, param in model.named_parameters():
        if param_check(name, param, args):
            print("[at weight {}]".format(name))
            print("percentage of pruned: {:.4f}%".format(100 * (abs(param) == 0).sum().item() / param.numel()))
            print("nonzero parameters after pruning: {} / {}\n".format((param != 0).sum().item(), param.numel()))
        total_param += param.numel()
        prune_param += (param != 0).sum().item()
    print("total nonzero parameters after pruning: {} / {} ({:.4f}%)".
          format(prune_param, total_param,
                 100 * (total_param - prune_param) / total_param))
