import torch
import os

from core.utils.parse_args import parse_args
from core.classifiers.load import init_classifier
from core.Trainer import Trainer
from core.utils.saver import Saver
from core.utils.plotter import Plotter
from core.utils.image_saver import ImageSaver
from core.utils.checkpointer import Checkpointer
import core.utils.dist_utils as dist_utils
from core.loaders.retrieve_loaders import retrieve_training_loaders, get_num_classes
from core.models.load import load_model
from core.algorithms.alg_selector import alg_selector
from core.utils.logger import get_logger

def main(args):

    # torch.manual_seed(1)

    if args.distributed is True:
        dist_utils.setup_dist_training(args)

    train_ldr, train_smp, val_ldr, val_smp, test_ldr, test_smp = retrieve_training_loaders(args)

    G = load_model(args.model_path, args.half_prec, args.munit_config_path)
    img_saver = ImageSaver(args, train_ldr, val_ldr, test_ldr, G=G, delta_dim=args.delta_dim)
    img_saver.save_images()
    
    model, opt, criterion, scheduler = init_classifier(args, get_num_classes(args.dataset))
    algorithm = alg_selector(model, criterion, G, args)

    if args.train_alg.lower() == 'mbst-primal-dual':
        dual_var = torch.tensor(1.0).cuda().requires_grad_(False)

    save_fname = algorithm.fname

    logger = get_logger(save_fname, args)
    if should_print(args):
        logger.info(f'Global fname: {save_fname}\n')
    
    trainer = Trainer(model, criterion, opt, logger, args)
    saver = Saver(save_fname, args)
    plotter = Plotter(save_fname, args, get_num_classes(args.dataset))
    checkpointer = Checkpointer(model, saver, save_fname, logger, args)

    for epoch in range(args.n_epochs):

        if args.distributed is True:
            train_smp.set_epoch(epoch)
            val_smp.set_epoch(epoch) if val_smp is not None else None
            test_smp.set_epoch(epoch)

        if args.train_alg.lower() == 'mbst-primal-dual':
            train_loss, train_acc, dual_var = trainer.train_primal_dual(train_ldr, epoch, alg=algorithm, dual_var=dual_var)
        else:
            train_loss, train_acc = trainer.train(train_ldr, epoch, alg=algorithm)
        valid_loss, valid_acc, valid_logits = trainer.evaluate(val_ldr)
        test_loss, test_acc, test_logits = trainer.evaluate(test_ldr)

        checkpointer.attempt_save(train_acc, valid_acc, test_acc, epoch)
        saver.update(train_loss, train_acc, test_loss, test_acc, valid_loss, valid_acc,
                        valid_logits, test_logits, None)
        plotter.plot(saver.df, None, None)

        if should_print(args) is True:
            logger.info(f'Train loss: {train_loss:.3f}\t Train accuracy: {train_acc:.3f}')
            if val_ldr is not None:
                logger.info(f'Valid loss: {valid_loss:.3f}\t Valid accuracy: {valid_acc:.3f}')
            logger.info(f'Test loss: {test_loss:.3f}\t Test accuracy: {test_acc:.3f}\n')

        if scheduler is not None:
            scheduler.step()

def should_print(args):
    if args.distributed is True and args.local_rank != 0:
        return False
    return True


if __name__ == '__main__':
    args = parse_args()
    main(args)