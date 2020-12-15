import torch
import time
from general_functions.utils import AverageMeter, save, accuracy
from architecture_functions.config_for_arch import CONFIG_ARCH

class TrainerArch:
    def __init__(self, criterion, optimizer, scheduler, logger, writer):
        self.top1   = AverageMeter()
        self.top3   = AverageMeter()
        self.losses = AverageMeter()
        
        self.logger = logger
        self.writer = writer
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        
        self.path_to_save_model = CONFIG_ARCH['train_settings']['path_to_save_model']
        self.cnt_epochs         = CONFIG_ARCH['train_settings']['cnt_epochs']
        self.print_freq         = CONFIG_ARCH['train_settings']['print_freq']
        
    def train_loop(self, train_loader, valid_loader, model):
        best_top1 = 0.0

        for epoch in range(self.cnt_epochs):
            
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            #if epoch and epoch % self.lr_decay_period == 0:
            #    self.optimizer.param_groups[0]['lr'] *= self.lr_decay

            # training
            self._train(train_loader, model, epoch)
            # validation
            top1_avg = self._validate(valid_loader, model, epoch)

            if best_top1 < top1_avg:
                best_top1 = top1_avg
                self.logger.info("Best top1 accuracy by now. Save model")
                save(model, self.path_to_save_model)
            self.scheduler.step()
        
    
    def _train(self, loader, model, epoch):
        start_time = time.time()
        model = model.train()

        for step, (X, y) in enumerate(loader):
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            #X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.shape[0]

            self.optimizer.zero_grad()
            outs = model(X)
            loss = self.criterion(outs, y)
            loss.backward()
            self.optimizer.step()

            self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Train")

        self._epoch_stats_logging(start_time=start_time, epoch=epoch, val_or_train='train')
        for avg in [self.top1, self.top3, self.losses]:
            avg.reset()

    def _validate(self, loader, model, epoch):
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.cuda(), y.cuda()
                N = X.shape[0]

                outs = model(X)
                loss = self.criterion(outs, y)
                
                self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Valid")
        
        top1_avg = self.top1.get_avg()
        self._epoch_stats_logging(start_time=start_time, epoch=epoch, val_or_train='val')
        for avg in [self.top1, self.top3, self.losses]:
            avg.reset()
        return top1_avg

    def _epoch_stats_logging(self, start_time, epoch, val_or_train):
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_loss', self.losses.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_top1', self.top1.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_top3', self.top3.get_avg(), epoch)
        
        top1_avg = self.top1.get_avg()
        self.logger.info(val_or_train + ": [{:3d}/{}] Final Prec@1 {:.4%} Time {:.2f}".format(
            epoch+1, self.cnt_epochs, top1_avg, time.time() - start_time))
        
    def _intermediate_stats_logging(self, outs, y, loss, step, epoch, N, len_loader, val_or_train):
        prec1, prec3 = accuracy(outs, y, topk=(1, 3))
        self.losses.update(loss.item(), N)
        self.top1.update(prec1.item(), N)
        self.top3.update(prec3.item(), N)
        
        if (step > 1 and step % self.print_freq == 0) or step == len_loader - 1:
            self.logger.info(val_or_train+
               ": [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} "
               "Prec@(1,3) ({:.1%}, {:.1%})".format(
                   epoch + 1, self.cnt_epochs, step, len_loader - 1, self.losses.get_avg(),
                   self.top1.get_avg(), self.top3.get_avg()))