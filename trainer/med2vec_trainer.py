#################################################################
# Code written by Sajad Darabi (sajad.darabi@cs.ucla.edu)
# For bug report, please contact author using the email address
#################################################################

import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer


class Med2VecTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Med2VecTrainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = config['trainer'].get('log_step', int(np.sqrt(data_loader.batch_size)))

    def _eval_metrics(self, output, target, **kwargs):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target, **kwargs)
            self.logger.info(f'{metric.__name__} {acc_metrics[i]}')
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()  # PyTorch的设计，要设定model的模式，来开启dropout等特殊层
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (x, ivec, jvec, mask, d) in enumerate(self.data_loader):
            data, ivec, jvec, mask, d = x.to(self.device), ivec.to(self.device), jvec.to(self.device), mask.to(self.device), d.to(self.device)
            self.optimizer.zero_grad()
            probits, emb_w = self.model(data.float(), d)  # 每个visit的预测输出
            # 计算输出到周围visit的loss TODO med2vec_loss
            loss_dict = self.loss(data, mask.float(), probits, self.model.bce_loss, emb_w, ivec, jvec, window=self.config["loss_window"])
            loss = loss_dict['visit_loss'] + loss_dict['code_loss']  # 不同级别的loss相加
            loss.backward()  # 前馈计算梯度
            self.optimizer.step()  # 更新参数

            # self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            # 记录结果
            self.logger.info(f'train: loss {loss.item()}, visit loss {loss_dict["visit_loss"]}, code loss {loss_dict["code_loss"]}')
            total_metrics += self._eval_metrics(probits.detach(), data.detach(), mask=mask,
                                                **self.config['trainer']['metrics_args'])  # TODO 实现不同metric不同的args

            self.logger.info(f'batch {batch_idx} in epoch {epoch}...')
            if self.verbosity >= 2 and (batch_idx % self.log_step == 0):
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] {}: {:.6f}, {}: {:.6f}'.format(
                    epoch,
                    batch_idx * self.data_loader.batch_size,
                    self.data_loader.n_samples,
                    100.0 * batch_idx / len(self.data_loader),
                    'visit_loss', loss_dict['visit_loss'],
                    'code_loss', loss_dict['code_loss']))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            total_loss += loss_dict['visit_loss'].detach() + loss_dict['code_loss'].detach()

            # if batch_idx == 10:
            #     break

        log = {
            'loss': total_loss / len(self.data_loader),  # log 一个epoch的平均 loss
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        """调整 learning rate"""
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (x, ivec, jvec, mask, d) in enumerate(self.valid_data_loader):
                self.logger.info(f'batch {batch_idx} in validation')
                data, ivec, jvec, mask, d = x.to(self.device), ivec.to(self.device), jvec.to(self.device), mask.to(self.device), d.to(self.device)
                probits, emb_w = self.model(data.float(), d)
                loss_dict = self.loss(data, mask.float(), probits, self.model.bce_loss, emb_w, ivec, jvec, window=self.config["loss_window"])
                loss = loss_dict['visit_loss'] + loss_dict['code_loss']
                self.logger.info(f'valid: loss {loss}, visit loss {loss_dict["visit_loss"]}, code loss {loss_dict["code_loss"]}')
                # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                # self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(probits.detach(), data.detach(), mask=mask,
                                                        **self.config['trainer']['metrics_args'])
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                if batch_idx==10:
                    break

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
