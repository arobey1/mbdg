import pandas as pd
import numpy as np
import os

class Saver:
    def __init__(self, fname, args):
        self._fname = fname + '.pd'
        self._n_epochs = args.n_epochs
        self._train_alg = args.train_alg
        self._args = args

        self._modes = ['Train', 'Validation', 'Test']
        self._data = {mode: {'loss': [], 'acc': [], 'logits': [], 'dists': []} for mode in self._modes}

        
        self._root = os.path.join(args.results_path, 'dataframes', args.train_alg)
        os.makedirs(self._root, exist_ok=True)

        self._logit_root = os.path.join(self._args.results_path, 'logit-dataframes', args.train_alg)
        os.makedirs(self._logit_root, exist_ok=True)

        self._dist_root = os.path.join(self._args.results_path, 'dist-dataframes', args.train_alg)
        os.makedirs(self._dist_root, exist_ok=True)

    @property
    def df(self):
        return self._df

    @property
    def logits_df(self):
        return self._logits_df

    @property
    def dist_df(self):
        return self._dist_df

    def __list_epochs(self):
        return range(self._n_epochs)

    def __to_list(self, var):
        return [var for _ in self.__list_epochs()]

    def is_best(self, acc, mode):
        if len(self._data[mode]['acc']) == 0:
            return True
        elif acc > max(self._data[mode]['acc']):
            return True
        return False

    def update(self, train_loss, train_acc, test_loss, test_acc, valid_loss, valid_acc,
                    valid_logits, test_logits, train_dists):
        self._data['Train']['loss'].append(train_loss)
        self._data['Train']['acc'].append(train_acc)
        self._data['Train']['dists'].append(train_dists)

        self._data['Validation']['loss'].append(valid_loss)
        self._data['Validation']['acc'].append(valid_acc)
        self._data['Validation']['logits'].append(valid_logits)

        self._data['Test']['loss'].append(test_loss)
        self._data['Test']['acc'].append(test_acc)
        self._data['Test']['logits'].append(test_logits)

        self.save()
        # self.save_logits()
        # self.save_dists()

    def save(self):

        epochs = self.__list_epochs()
        train_alg = self.__to_list(self._args.train_alg)
        lr = self.__to_list(self._args.lr)
        trial = self.__to_list(self._args.trial_index)
        arch = self.__to_list(self._args.architecture)
        default_cols = ['Epoch', 'Training-Alg', 'Mode', 'Learning-Rate',
                            'Trial', 'Architecture', 'Accuracy', 'Loss']

        all_dfs = []
        for mode in self._modes:
            default_data = [epochs, train_alg, self.__to_list(mode), lr, trial, arch]
            default_data.extend([self._data[mode]['acc'], self._data[mode]['loss']])

            if self._train_alg.lower() == 'mbdg-reg':
                df = self.__get_mbdg_reg_df(default_cols, default_data)
            elif self._train_alg.lower() == 'erm':
                df = self.__get_erm_df(default_cols, default_data)
            elif self._train_alg.lower() == 'mbdg-primal-dual':
                df = self.__get_mbdg_df(default_cols, default_data)
            else:
                raise ValueError()

            all_dfs.append(df)

        self._df = pd.concat(all_dfs, ignore_index=True)
        self._df.to_pickle(os.path.join(self._root, self._fname))
            

    def save_logits(self):

        def create_df(array, epoch, mode):
            num_datapoints = array.shape[0]
            sorted_logits = np.sort(array, axis=1)
            flat_logits = sorted_logits.flatten()[::-1]
            indices = np.tile(np.arange(2), num_datapoints)
            df = pd.DataFrame(list(zip(indices, flat_logits)), columns=['Sorted-Indices', 'Logit-Vals'])
            df['Epoch'] = epoch
            df['Mode'] = mode
            df['Training-Alg'] = self._args.train_alg
            df['Trial'] = self._args.trial_index
            return df

        all_dfs = []
        for mode in ['Validation', 'Test']:
            for epoch, array in enumerate(self._data[mode]['logits']):
                df = create_df(array, epoch, mode)
                all_dfs.append(df)

        self._logits_df = pd.concat(all_dfs, ignore_index=True)
        fname = os.path.join(self._logit_root, self._fname)
        self._logits_df.to_pickle(fname)

    def save_dists(self):

        def create_df(array, epoch):
            df = pd.DataFrame(array, columns=['Distances'])
            df['Epoch'] = epoch
            df['Mode'] = 'Train'
            df['Training-Alg'] = self._args.train_alg
            df['Trial'] = self._args.trial_index
            return df

        all_dfs = []
        for epoch, array in enumerate(self._data['Train']['dists']):
            df = create_df(array, epoch)
            all_dfs.append(df)

        self._dist_df = pd.concat(all_dfs, ignore_index=True)
        fname = os.path.join(self._dist_root, self._fname)
        self._dist_df.to_pickle(fname)

    def __get_mbdg_reg_df(self, default_cols, default_data):

        lam_dist = self.__to_list(self._args.mbdg_static_lam_dist)
        lam_grad = self.__to_list(self._args.mbdg_static_lam_grad)
        num_steps = self.__to_list(self._args.mbdg_num_steps)

        cols = default_cols + ['Lambda-Dist', 'Lambda-Grad', 'Num-Steps']
        data = list(zip(*default_data, lam_dist, lam_grad, num_steps))
        return pd.DataFrame(data, columns=cols)

    def __get_mbdg_df(self, default_cols, default_data):

        dual_step_size = self.__to_list(self._args.mbdg_dual_step_size)
        num_steps = self.__to_list(self._args.mbdg_num_steps)
        gamma = self.__to_list(self._args.mbdg_gamma)

        cols = default_cols + ['Dual-Step-Size', 'Gamma-Margin', 'Num-Steps']
        data = list(zip(*default_data, dual_step_size, gamma, num_steps))
        return pd.DataFrame(data, columns=cols)

    def __get_erm_df(self, default_cols, default_data):
        return pd.DataFrame(list(zip(*default_data)), columns=default_cols)
