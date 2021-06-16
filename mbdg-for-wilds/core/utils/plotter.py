import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import os

class Plotter:
    def __init__(self, fname, args, n_classes):
        self._fname = fname + '.png'
        self._args = args
        self._n_classes = n_classes
        
        root = os.path.join(args.results_path, 'plots', args.train_alg)
        os.makedirs(root, exist_ok=True)

        self._acc_root = os.path.join(root, 'accuracy')
        os.makedirs(self._acc_root, exist_ok=True)

        self._logits_root = os.path.join(root, 'logits')
        os.makedirs(self._logits_root, exist_ok=True)

        self._stab_root = os.path.join(root, 'stability')
        os.makedirs(self._stab_root, exist_ok=True)

    def plot(self, df, logits_df, dist_df):

        self.__plot_acc(df)
        # self.__plot_logits(logits_df)
        # self.__plot_dist(dist_df)

    def __plot_dist(self, dist_df):
        g = sns.FacetGrid(data=dist_df, col='Epoch', col_wrap=5, height=3)
        g.map_dataframe(sns.boxplot, x='Training-Alg', y='Distances')
        plt.tight_layout()

        fname = os.path.join(self._stab_root, self._fname)
        plt.savefig(fname)
        plt.close()
        
    def __plot_logits(self, logits_df):

        g = sns.FacetGrid(data=logits_df, col='Epoch', col_wrap=5, height=3)
        g.map_dataframe(sns.boxplot, x='Sorted-Indices', y='Logit-Vals', order=range(self._n_classes))
        plt.tight_layout()

        fname = os.path.join(self._logits_root, self._fname)
        plt.savefig(fname)
        plt.close()

    def __plot_acc(self, df):

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

        sns.set(style='darkgrid', font='Palatino')
        kwargs = {'x': 'Epoch', 'data': df, 'linewidth': 3, 'hue': 'Mode'}

        sns.lineplot(y='Loss', ax=ax2, **kwargs)
        sns.lineplot(y='Accuracy', ax=ax1, **kwargs)
        plt.tight_layout()
        
        fname = os.path.join(self._acc_root, self._fname)
        plt.savefig(fname)
        plt.close()

