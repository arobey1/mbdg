import os
import torch

class Checkpointer:

    def __init__(self, model, saver, fname, logger, args):
        self._model = model
        self._saver = saver
        self._logger = logger
        self._fname = fname
        self._args = args

        root = os.path.join(args.results_path, 'checkpoints', args.train_alg)
        os.makedirs(root, exist_ok=True)

        self._roots = {}
        for mode in ['Train', 'Val', 'Test']:
            mode_root = os.path.join(root, mode)
            os.makedirs(mode_root, exist_ok=True)
            self._roots[mode] = mode_root

    def attempt_save(self, train_acc, val_acc, test_acc, epoch):

        if self._saver.is_best(train_acc, 'Train') is True:
            self.save_ckpt(self._roots['Train'])
            if self.should_print() is True:
                self._logger.info(f'Saving best TRAINING accuracy model @ epoch {epoch}')
        
        if self._saver.is_best(val_acc, 'Validation') is True:
            self.save_ckpt(self._roots['Val'])
            if self.should_print() is True:
                self._logger.info(f'Saving best VALIDATION accuracy model @ epoch {epoch}')

        if self._saver.is_best(test_acc, 'Test') is True:
            self.save_ckpt(self._roots['Test'])
            if self.should_print() is True:
                self._logger.info(f'Saving best TEST accuracy model @ epoch {epoch}')

    def save_ckpt(self, root):
        fname = os.path.join(root, f'{self._fname} + .pth')
        torch.save(self._model.module.state_dict(), fname)

    def should_print(self):
        if self._args.distributed is True and self._args.local_rank != 0:
            return False
        return True
