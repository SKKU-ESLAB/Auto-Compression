import logging
from pathlib import Path

import torch as t
import yaml

import process_tune
import quan
import util
from model import create_model
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

script_dir = Path.cwd()
args = util.get_config(default_file=script_dir / 'config.yaml')

output_dir = script_dir / args.output_dir
output_dir.mkdir(exist_ok=True)

log_dir = util.init_logger(args.name, output_dir, script_dir / 'logging.conf')
logger = logging.getLogger()

with open(log_dir / "args.yaml", "w") as yaml_file:  # dump experiment config
    yaml.safe_dump(args, yaml_file)

def train(config):
    import torch as t
    pymonitor = util.ProgressMonitor(logger)
    tbmonitor = util.TensorBoardMonitor(logger, log_dir)
    monitors = [pymonitor, tbmonitor]

    if args.device.type == 'cpu' or not t.cuda.is_available() or args.device.gpu == []:
        args.device.gpu = []
    else:
        available_gpu = t.cuda.device_count()
        for dev_id in args.device.gpu:
            if dev_id >= available_gpu:
                logger.error('GPU device ID {0} requested, but only {1} devices available'
                             .format(dev_id, available_gpu))
                exit(1)
        # Set default device in case the first one on the list
        t.cuda.set_device(args.device.gpu[0])
        # Enable the cudnn built-in auto-tuner to accelerating training, but it
        # will introduce some fluctuations in a narrow range.
        t.backends.cudnn.benchmark = True
        t.backends.cudnn.deterministic = False

    # Initialize data loader
    train_loader, val_loader, test_loader = util.load_data(args.dataloader)
    logger.info('Dataset `%s` size:' % args.dataloader.dataset +
                '\n          Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
                '\n        Validation Set = %d (%d)' % (len(val_loader.sampler), len(val_loader)) +
                '\n              Test Set = %d (%d)' % (len(test_loader.sampler), len(test_loader)))

    print('Dataset `%s` size:' % args.dataloader.dataset +
                '\n          Training Set = %d (%d)' % (len(train_loader.sampler), len(train_loader)) +
                '\n        Validation Set = %d (%d)' % (len(val_loader.sampler), len(val_loader)) +
                '\n              Test Set = %d (%d)' % (len(test_loader.sampler), len(test_loader)))
    # Create the model
    model = create_model(args)
    modules_to_replace = quan.find_modules_to_quantize(model, args.quan)
    model = quan.replace_module_by_names(model, modules_to_replace)
    tbmonitor.writer.add_graph(model, input_to_model=train_loader.dataset[0][0].unsqueeze(0))
    logger.info('Inserted quantizers into the original model')
    print('Inserted quantizers into the original model')
    if args.device.gpu and not args.dataloader.serialized:
        model = t.nn.DataParallel(model, device_ids=args.device.gpu)
    
    model.to(args.device.type)

    start_epoch = 0
    if args.resume.path:
        model, start_epoch, _ = util.load_checkpoint(
            model, args.resume.path, args.device.type, lean=args.resume.lean)

    # Define loss function (criterion) and optimizer
    criterion = t.nn.CrossEntropyLoss().to(args.device.type)

    # optimizer = t.optim.Adam(model.parameters(), lr=args.optimizer.learning_rate)
    optimizer = t.optim.SGD(model.parameters(),
                            lr=args.optimizer.learning_rate,
                            momentum=args.optimizer.momentum,
                            weight_decay=args.optimizer.weight_decay)
    lr_scheduler = util.lr_scheduler(optimizer,
                                     batch_size=train_loader.batch_size,
                                     num_samples=len(train_loader.sampler),
                                     **args.lr_scheduler)
    logger.info(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    print(('Optimizer: %s' % optimizer).replace('\n', '\n' + ' ' * 11))
    logger.info('LR scheduler: %s\n' % lr_scheduler)
    print('LR scheduler: %s\n' % lr_scheduler)

    perf_scoreboard = process_tune.PerformanceScoreboard(args.log.num_best_scores)

    if args.eval:
        process_tune.validate(test_loader, model, criterion, -1, monitors, args)
    else:  # training
        if args.resume.path or args.pre_trained:
            logger.info('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
            print('>>>>>>>> Epoch -1 (pre-trained model evaluation)')
            top1, top5, _,sparsity = process_tune.validate(val_loader, model, criterion,
                                             start_epoch - 1, monitors, args)
            perf_scoreboard.update(top1, top5, start_epoch - 1, sparsity)
        for epoch in range(start_epoch, args.epochs):
            logger.info('>>>>>>>> Epoch %3d' % epoch)
            print('>>>>>>>> Epoch %3d' % epoch)
            t_top1, t_top5, t_loss = process_tune.train(train_loader, model, criterion, optimizer,
                                                   lr_scheduler, epoch, monitors, config['alpha'],args)
            v_top1, v_top5, v_loss,sparsity = process_tune.validate(val_loader, model, criterion, epoch, monitors, args)

            tbmonitor.writer.add_scalars('Train_vs_Validation/Loss', {'train': t_loss, 'val': v_loss}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top1', {'train': t_top1, 'val': v_top1}, epoch)
            tbmonitor.writer.add_scalars('Train_vs_Validation/Top5', {'train': t_top5, 'val': v_top5}, epoch)

            perf_scoreboard.update(v_top1, v_top5, epoch, sparsity)
            is_best = perf_scoreboard.is_best(epoch)

            util.save_checkpoint(epoch, args.arch, model, {'top1': v_top1, 'top5': v_top5}, is_best, args.name, log_dir)
            
            tune.report(v_top1 = v_top1, v_top5 = v_top5,loss = v_loss, sparsity = sparsity.detach().cpu(), v_top1_sparsity = (v_top1-87)/10 + sparsity.detach().cpu() * 0.8)
        logger.info('>>>>>>>> Epoch -1 (final model evaluation)')
        print('>>>>>>>> Epoch -1 (final model evaluation)')
        process_tune.validate(test_loader, model, criterion, -1, monitors, args)
    tbmonitor.writer.close()  # close the TensorBoard
    logger.info('Program completed successfully ... exiting ...')
    logger.info('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')
    print('Program completed successfully ... exiting ...')
    print('If you have any questions or suggestions, please visit: github.com/zhutmost/lsq-net')

def main(num_samples = 10, max_num_epochs = 10, gpus_per_trial = 2):
    config = {
            "alpha" : tune.grid_search([1e-9, 5e-5,1e-4, 5e-5, 1e-5, 1e-6, 1e-7, 1e-8])
            }

    scheduler = ASHAScheduler(
            metric = "v_top1_sparsity",
            mode = "max",
            max_t = max_num_epochs,
            grace_period = 90,
            reduction_factor = 2)
    reporter = CLIReporter(
            metric_columns = ["loss", "v_top1", "v_top5","sparsity", "v_top1_sparsity","training_iteration"])

    result = tune.run(
            train,
            resources_per_trial = {"cpu": 16, "gpu": gpus_per_trial},
            config = config,
            num_samples = 1,
            scheduler = scheduler,
            progress_reporter = reporter)
    
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
    best_trial.last_result["v_top1"]))
    print("Best trial final validation accuracy5: {}".format(
    best_trial.last_result["v_top5"]))
    print("Best trial final validation sparsity: {}".format(
    best_trial.last_result["sparsity"]))

    

if __name__ == "__main__":
    main(num_samples = 10, max_num_epochs = args.epochs, gpus_per_trial = 2)

