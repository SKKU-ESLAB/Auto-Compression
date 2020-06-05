import torch
from torch.autograd import Variable
import time
from general_functions.utils import AverageMeter, save, accuracy
from supernet_functions.config_for_supernet import CONFIG_SUPERNET

class TrainerSupernet:
    def __init__(self, criterion, w_optimizer, theta_optimizer, w_scheduler, logger, writer, high):
        self.top1       = AverageMeter()
        self.top5       = AverageMeter()
        self.losses     = AverageMeter()
        self.losses_lat = AverageMeter()
        self.losses_ce  = AverageMeter()
        
        self.logger = logger
        self.writer = writer
        
        self.criterion = criterion
        self.w_optimizer = w_optimizer
        self.theta_optimizer = theta_optimizer
        self.w_scheduler = w_scheduler
        self.high = high

        self.temperature                 = CONFIG_SUPERNET['train_settings']['init_temperature']
        self.exp_anneal_rate             = CONFIG_SUPERNET['train_settings']['exp_anneal_rate'] # apply it every epoch
        self.cnt_epochs                  = CONFIG_SUPERNET['train_settings']['cnt_epochs']
        self.train_thetas_from_the_epoch = CONFIG_SUPERNET['train_settings']['train_thetas_from_the_epoch']
        self.print_freq                  = CONFIG_SUPERNET['train_settings']['print_freq']
        if high:
            self.path_to_save_model          = CONFIG_SUPERNET['train_settings']['path_to_save_model_high']
        else:
            self.path_to_save_model          = CONFIG_SUPERNET['train_settings']['path_to_save_model']
    
    def train_loop(self, train_w_loader, train_thetas_loader, test_loader, model):
        
        best_top1 = 0.0
        
        # firstly, train weights only
        for epoch in range(self.train_thetas_from_the_epoch):
            self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
            
            self.logger.info("Firstly, start to train weights for epoch %d" % (epoch))
            #self._training_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")
            #self.w_scheduler.step()

            top1_avg = self._validate(model, test_loader, epoch)
            if best_top1 < top1_avg:
                best_top1 = top1_avg
                self.logger.info("Best top1 acc by now. Save model")
                save(model, self.path_to_save_model)
                print("Best model saved!")
        
        for epoch in range(self.train_thetas_from_the_epoch, self.cnt_epochs):
            self.writer.add_scalar('learning_rate/weights', self.w_optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('learning_rate/theta', self.theta_optimizer.param_groups[0]['lr'], epoch)
            
            self.logger.info("Start to train weights for epoch %d" % (epoch))
            self._training_step(model, train_w_loader, self.w_optimizer, epoch, info_for_logger="_w_step_")
            self.w_scheduler.step()
            
            self.logger.info("Start to train theta for epoch %d" % (epoch))
            self._training_step(model, train_thetas_loader, self.theta_optimizer, epoch, info_for_logger="_theta_step_")
            
            top1_avg = self._validate(model, test_loader, epoch)
            if best_top1 < top1_avg:
                best_top1 = top1_avg
                self.logger.info("Best top1 acc by now. Save model")
                save(model, self.path_to_save_model)
                print("Best model saved!")
            
            self.temperature = self.temperature * self.exp_anneal_rate
       
    def _training_step(self, model, loader, optimizer, epoch, info_for_logger=""):
        model = model.train()
        start_time = time.time()
        
        for step, (X, y) in enumerate(loader):
            X, y = X.cuda(non_blocking=True), y.cuda(non_blocking=True)
            # X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.shape[0]
            optimizer.zero_grad()
            #latency_to_accumulate = Variable(torch.Tensor([[0.0]]), requires_grad=True).cuda()
            #outs, latency_to_accumulate = model(X, self.temperature, latency_to_accumulate)
            outs, latency_to_accumulate = model(X, self.temperature)
            latency_to_accumulate = torch.sum(latency_to_accumulate)
            loss = self.criterion(outs, y, latency_to_accumulate, self.losses_ce, self.losses_lat, N)
            loss.backward()
            optimizer.step()
            
            self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Train")
        
        self._epoch_stats_logging(start_time=start_time, epoch=epoch, info_for_logger=info_for_logger, val_or_train='train')
        for avg in [self.top1, self.top5, self.losses]:
            avg.reset()
        
    def _validate(self, model, loader, epoch):
        model.eval()
        start_time = time.time()

        with torch.no_grad():
            for step, (X, y) in enumerate(loader):
                X, y = X.cuda(), y.cuda()
                N = X.shape[0]
                
                #latency_to_accumulate = torch.Tensor([[0.0]]).cuda()
                #outs, latency_to_accumulate = model(X, self.temperature, latency_to_accumulate)
                outs, latency_to_accumulate = model(X, self.temperature)
                latency_to_accumulate = torch.sum(latency_to_accumulate)
                loss = self.criterion(outs, y, latency_to_accumulate, self.losses_ce, self.losses_lat, N)

                self._intermediate_stats_logging(outs, y, loss, step, epoch, N, len_loader=len(loader), val_or_train="Valid")
                
        top1_avg = self.top1.get_avg()
        self._epoch_stats_logging(start_time=start_time, epoch=epoch, val_or_train='val')
        for avg in [self.top1, self.top5, self.losses]:
            avg.reset()
        return top1_avg
    
    def _epoch_stats_logging(self, start_time, epoch, val_or_train, info_for_logger=''):
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_loss'+info_for_logger, self.losses.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_top1'+info_for_logger, self.top1.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_top5'+info_for_logger, self.top5.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_losses_lat'+info_for_logger, self.losses_lat.get_avg(), epoch)
        self.writer.add_scalar('train_vs_val/'+val_or_train+'_losses_ce'+info_for_logger, self.losses_ce.get_avg(), epoch)
        
        top1_avg = self.top1.get_avg()
        self.logger.info(info_for_logger+val_or_train + ": [{:3d}/{}] Final Prec@1 {:.4%} Time {:.2f}".format(
            epoch+1, self.cnt_epochs, top1_avg, time.time() - start_time))
        
    def _intermediate_stats_logging(self, outs, y, loss, step, epoch, N, len_loader, val_or_train):
        prec1, prec5 = accuracy(outs, y, topk=(1, 5))
        self.losses.update(loss.item(), N)
        self.top1.update(prec1.item(), N)
        self.top5.update(prec5.item(), N)
        
        if (step > 1 and step % self.print_freq == 0) or step == len_loader - 1:
            self.logger.info(val_or_train+
               ": [{:3d}/{}] Step {:03d}/{:03d} Loss {:.3f} "
               "Prec@(1,5) ({:.1%}, {:.1%}), ce_loss {:.3f}, lat_loss {:.3f}".format(
                   epoch + 1, self.cnt_epochs, step, len_loader - 1, self.losses.get_avg(),
                   self.top1.get_avg(), self.top5.get_avg(), self.losses_ce.get_avg(), self.losses_lat.get_avg()))
        
