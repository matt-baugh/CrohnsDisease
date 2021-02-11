import os
from sklearn.metrics import f1_score
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

from train_util import report
from mri_dataset import MRIDataset

USE_GPU = True


class PytorchTrainer:
    def __init__(self, args, model):
        # Paths
        self.args = args
        self.logdir = os.path.join(args.base, args.logdir)
        self.fold = args.fold
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.model_save_path = os.path.join(args.base, args.model_path)

        self.train_data_path = os.path.join(args.base, args.train_datapath)
        self.test_data_path = os.path.join(args.base, args.test_datapath)

        self.train_dataset = MRIDataset(self.train_data_path, True, args.feature_shape)
        self.test_dataset = MRIDataset(self.test_data_path, False, args.feature_shape)

        self.summary = SummaryWriter(self.logdir, f'fold{self.fold}')

        self.write_log(f'Fold: {self.fold}')

        if USE_GPU and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Data processing
        self.record_shape = args.record_shape

        # General parameters
        self.test_evaluation_period = 1
        self.num_batches = int(args.num_batches)

        # Network parameters
        self.model = model
        self.attention = args.attention
        self.feature_shape = args.feature_shape
        self.batch_size = args.batch_size
        self.test_size = min(self.batch_size, len(self.test_dataset))

        # Hyperparameters
        self.weight_decay = 0# 1e-4
        self.dropout_train_prob = 0.5
        starter_learning_rate = 5e-6
        self.learning_rate = starter_learning_rate

        # Logging
        self.best = {'batch': None, 'report': None, 'preds': None, 'loss': float("inf")}

    def write_log(self, line):
        self.summary.add_text('Log', line)

    def update_stats(self, batch, loss, preds, labels):
        if loss < self.best['loss']:
            self.best['batch'] = batch
            self.best['loss'] = loss
            self.best['preds'] = preds
            self.best['labels'] = labels
            self.best['report'] = report(labels, preds)

    def train(self):

        train_loader = DataLoader(self.train_dataset, self.batch_size, True)
        test_loader = DataLoader(self.test_dataset, self.test_size, False)

        train_step = 0

        network = self.model
        optimiser = Adam(network.parameters(), lr=self.learning_rate)

        train_accuracies = []
        while train_step <= self.num_batches:

            for (x, y) in train_loader:
                network.train()

                x.to(device=self.device)
                y.to(device=self.device)
                binary_y = torch.where(y == 0, 0, 1)

                out = network(x)
                preds = out.argmax(dim=1)

                loss = F.cross_entropy(out, y)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # Summaries and statistics
                print(f'-- Train Batch {train_step} --')

                self.summary.add_scalar('Loss/train', loss.item(), train_step)

                train_accuracies.append((preds == binary_y).mean())
                running_accuracy = torch.mean(torch.stack(train_accuracies[-self.test_evaluation_period:]))
                self.summary.add_scalar('Accuracy/train', running_accuracy, train_step)

                train_f1 = f1_score(binary_y, preds)
                self.summary.add_scalar('F1 Score/train', train_f1, train_step)

                print('Loss:               ', loss)
                print('Prediction balance: ', preds.mean())
                print(report(binary_y, preds))

                train_step += 1
