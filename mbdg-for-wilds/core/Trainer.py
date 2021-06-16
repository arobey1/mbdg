import torch
from apex import amp
import torch.nn.functional as F

import core.utils.dist_utils as dist_utils
from core.utils.meters import AverageMeter, TimeMeter, VarTracker

class Trainer:

    def __init__(self, model, criterion, optimizer, logger, args):
        """Constructor for trainer."""

        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._logger = logger
        self._args = args

    def train_primal_dual(self, loader, epoch, alg, dual_var):

        acc_meter, loss_meter = AverageMeter(), AverageMeter()
        dual_meter = AverageMeter()
        timer = TimeMeter()

        self._model.train()
        for batch, (imgs, labels) in enumerate(loader):
            timer.batch_start()
            imgs, labels = imgs.cuda(), labels.cuda()
            
            self._optimizer.zero_grad()
            loss, correct, reg_term = alg(imgs, labels, dual_var)

            if self._args.half_prec is True:
                with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            self._optimizer.step()

            batch_size = imgs.size(0)
            if self._args.distributed is True:
                metrics = torch.tensor([loss.item(), correct, batch_size, reg_term.item()]).float().cuda()
                reduced_loss, correct, batch_size, reg_term = dist_utils.sum_tensor(metrics).cpu().numpy()
                loss = reduced_loss / dist_utils.env_world_size()
                reg_term = reg_term / dist_utils.env_world_size()

            acc = 100. * correct / batch_size
            dual_var = alg.dual_step(dual_var, reg_term)
            dual_meter.update(dual_var.item(), n=1)

            acc_meter.update(acc.item(), n=batch_size)
            loss_meter.update(loss.item(), n=batch_size)
            timer.batch_end()

            if self._should_print(batch % self._args.log_interval == 0):
                pct = 100. * batch / len(loader)
                out = (f'Train epoch: {epoch} [{batch}/{len(loader)} ({pct:.0f}%)]' + 
                        f'\tLoss: {loss_meter.val:.3f} (avg: {loss_meter.avg:.3f})' + 
                        f'\tTime: {timer.batch_time.val:.3f} (avg. {timer.batch_time.avg:.3f})' + 
                        f'\tDual variable {dual_meter.val:.3f} (avg. {dual_meter.avg:.3f})')
                self._logger.info(out)
                
        return loss_meter.avg, acc_meter.avg, dual_var

    def train(self, loader, epoch, alg):
        """Train for one epoch to minimize the empirical risk of the classifier.
        
        Params:
            loader: DataLoader for inner descent step.
            epoch: Current training epoch.
        """

        acc_meter, loss_meter = AverageMeter(), AverageMeter()
        timer = TimeMeter()

        self._model.train()
        for batch, (imgs, labels) in enumerate(loader):
            timer.batch_start()
            imgs, labels = imgs.cuda(), labels.cuda()
            
            self._optimizer.zero_grad()
            loss, correct = alg(imgs, labels)

            if self._args.half_prec is True:
                with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            self._optimizer.step()

            batch_size = imgs.size(0)
            if self._args.distributed is True:
                metrics = torch.tensor([loss.item(), correct, batch_size]).float().cuda()
                reduced_loss, correct, batch_size = dist_utils.sum_tensor(metrics).cpu().numpy()
                loss = reduced_loss / dist_utils.env_world_size()

            acc = 100. * correct / batch_size

            acc_meter.update(acc.item(), n=batch_size)
            loss_meter.update(loss.item(), n=batch_size)
            timer.batch_end()

            if self._should_print(batch % self._args.log_interval == 0):
                pct = 100. * batch / len(loader)
                out = (f'Train epoch: {epoch} [{batch}/{len(loader)} ({pct:.0f}%)]' + 
                        f'\tLoss: {loss_meter.val:.3f} (avg: {loss_meter.avg:.3f})' + 
                        f'\tTime: {timer.batch_time.val:.3f} (avg. {timer.batch_time.avg:.3f})')
                self._logger.info(out)

        return loss_meter.avg, acc_meter.avg

    @torch.no_grad()
    def evaluate(self, loader, alg=None):
        """Evaluate the performance of the classifier.
        
        Params:
            loader: DataLoader for evaluating the classifier.

        Returns:
            test_loss: Average empirical risk over the test set.
            acc: Accuracy of trained classifier.
        """

        if loader is None:
            return None, None

        acc_meter, loss_meter = AverageMeter(), AverageMeter()
        output_tracker = VarTracker()

        self._model.eval()

        for (imgs, labels) in loader:
            imgs, labels = imgs.cuda(), labels.cuda()

            if self._args.distributed is True:
                acc, loss, batch_total, output = self._distributed_predict(imgs, labels, alg=alg)
            else:
                output = self._model(imgs)
                pred = output.argmax(dim=1, keepdim=True)
                loss = self._criterion(output, labels)
                correct = pred.eq(labels.view_as(pred)).sum().item()
                batch_total = imgs.size(0)
                acc = 100. * correct / batch_total

            # output_tracker.update(F.softmax(output, dim=1).cpu().numpy())
            acc_meter.update(acc, n=batch_total)
            loss_meter.update(loss.item(), n=batch_total)

        # return loss_meter.avg, acc_meter.avg, output_tracker.stacked()
        return loss_meter.avg, acc_meter.avg, None

    def _should_print(self, cond=True):
        """Determines if a process should print."""

        if self._args.distributed is True:
            if cond is True and self._args.local_rank == 0:
                return  True
        elif cond is True:
            return True
        return False

    def _distributed_predict(self, imgs, labels, alg=None):
        """Makes predictions over distributed processes."""

        batch_size = imgs.size(0)
        output = loss = corr1 = valid_batches = 0

        if batch_size:
            if alg is not None:
                loss, corr = alg(imgs, labels)
                valid_batches = 1
            else:
                output = self._model(imgs)
                loss = self._criterion(output, labels).data
                pred = output.argmax(dim=1, keepdim=True)
                    
                # measure accuracy and record loss
                valid_batches = 1
                corr = pred.eq(labels.view_as(pred)).sum()

        metrics = torch.tensor([batch_size, valid_batches, loss, corr]).float().cuda()
        batch_total, valid_batches, reduced_loss, corr = dist_utils.sum_tensor(metrics).cpu().numpy()
        reduced_loss = reduced_loss / valid_batches
        top1 = corr * (100.0 / batch_total)

        # all_output = dist_utils.multi_varsize_all_gather(output, cat_dim=0)

        # return top1, reduced_loss, batch_total, all_output
        return top1, reduced_loss, batch_total, None