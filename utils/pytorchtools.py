import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



class double_path_EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path1='checkpoint1.pt', path2='checkpoint2.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score_1 = None
        self.best_score_2 = None
        self.early_stop = False
        self.val_loss_min_1 = np.Inf
        self.val_loss_min_2 = np.Inf

        self.delta = delta
        self.path1 = path1
        self.path2 = path2

        self.trace_func = trace_func
    def __call__(self, val_loss_1, val_loss_2, model1, model2):

        score1 = -val_loss_1
        score2 = -val_loss_2

        if self.best_score_1 is None:
            self.best_score_1 = score1
            self.save_checkpoint_path1(val_loss_1, model1)

        if self.best_score_2 is None:
            self.best_score_2 = score2
            self.save_checkpoint_path2(val_loss_2, model2)

        elif (score1 < self.best_score_1 + self.delta) \
                and (score2 < self.best_score_2 + self.delta):
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if score1 >= self.best_score_1 + self.delta:
                self.best_score_1 = score1
                self.save_checkpoint_path1(val_loss_1, model1)
            if score2 >= self.best_score_2 + self.delta:
                self.best_score_2 = score2
                self.save_checkpoint_path2(val_loss_2, model2)

            self.counter = 0

    def save_checkpoint_path1(self, val_loss, model1):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Model 1 Validation loss decreased ({self.val_loss_min_1:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model1.state_dict(), self.path1)
        self.val_loss_min_1 = val_loss

    def save_checkpoint_path2(self, val_loss, model2):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Model 2 Validation loss decreased ({self.val_loss_min_2:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model2.state_dict(), self.path2)
        self.val_loss_min_2 = val_loss


