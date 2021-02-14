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
    def __init__(self, args):
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
        self.attention = args.attention
        self.feature_shape = args.feature_shape
        self.batch_size = args.batch_size
        self.test_size = min(self.batch_size, len(self.test_dataset))

        # Hyperparameters
        self.dropout_train_prob = 0.5
        starter_learning_rate = 5e-6
        self.learning_rate = starter_learning_rate

        self.train_loader = DataLoader(self.train_dataset, self.batch_size, True)
        self.test_loader = DataLoader(self.test_dataset, self.test_size, False)

        # Best Test Results
        self.best = {'iteration': None,
                     'report': None,
                     'preds': None,
                     'labels': None,
                     'MaRIAs': None,
                     'loss': float("inf")}

    def write_log(self, line):
        self.summary.add_text('Log', line)

    def log_statistics(self, tag, loss, acc, f1, train_step):
        self.summary.add_scalar('Loss/' + tag, loss, train_step)
        self.summary.add_scalar('Accuracy/' + tag, acc, train_step)
        self.summary.add_scalar('F1 Score/' + tag, f1, train_step)

    def evaluate_on_test(self, network, train_step):

        all_binary_labels, all_preds, all_losses, all_y = [], [], [], []

        network.eval()
        for (x, y) in self.test_loader:

            x.to(device=self.device)
            y.to(device=self.device)
            binary_y = torch.where(y == 0, 0, 1)

            with torch.no_grad():
                out = network(x)
            preds = out.argmax(dim=1)

            loss = F.cross_entropy(out, binary_y)

            all_binary_labels.append(binary_y)
            all_preds.append(preds)
            all_losses += [loss] * len(y)
            all_y.append(y)

        all_binary_labels = torch.cat(all_binary_labels)
        all_preds = torch.cat(all_preds)
        all_losses = torch.cat(all_losses)
        all_y = torch.cat(all_y)

        test_avg_acc = (all_preds == all_binary_labels).mean()
        test_avg_loss = all_losses.mean()
        test_f1 = f1_score(all_binary_labels, all_preds)

        if test_avg_loss < self.best['loss']:

            self.best['iteration'] = train_step
            self.best['loss'] = test_avg_loss
            self.best['preds'] = all_preds
            self.best['labels'] = all_binary_labels
            self.best['MaRIAs'] = all_y
            self.best['report'] = report(all_binary_labels, all_preds)

            torch.save(network.state_dict(), self.model_save_path)
            print()
            print('===========================> Model (almost) saved!')
            print()

        print('Test statistics')
        print('Average Loss:       ', test_avg_loss)
        print('Prediction balance: ', all_preds.mean())
        print(report(all_binary_labels, all_preds))
        print()

        self.log_statistics('test', test_avg_loss, test_avg_acc, test_f1, train_step)

    def train(self):

        train_step = 0

        network = self.model
        # Attention
        # Feature shape ??
        # Dropout probability
        optimiser = Adam(network.parameters(), lr=self.learning_rate)

        train_accuracies = []
        while train_step <= self.num_batches:

            for (x, y) in self.train_loader:
                network.train()

                x.to(device=self.device)
                y.to(device=self.device)
                binary_y = torch.where(y == 0, 0, 1)

                out = network(x)
                preds = out.argmax(dim=1)

                loss = F.cross_entropy(out, binary_y)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # Summaries and statistics
                print(f'-- Train Batch {train_step} --')
                print('Loss:               ', loss)
                print('Prediction balance: ', preds.mean())
                print(report(binary_y, preds))
                print()

                train_accuracies.append((preds == binary_y).mean())
                running_accuracy = torch.mean(torch.stack(train_accuracies[-self.test_evaluation_period:]))
                train_f1 = f1_score(binary_y, preds)

                self.summary.add_scalar('Loss/train', loss.item(), train_step)
                self.summary.add_scalar('Accuracy/train', running_accuracy, train_step)
                self.summary.add_scalar('F1 Score/train', train_f1, train_step)

                if train_step % self.test_evaluation_period == 0:
                    self.evaluate_on_test(network, train_step)

                train_step += 1

        print('Training finished!')
        self.write_log(f'Best loss (epoch {self.best["batch"]}): {round(self.best["loss"], 3)}')
        self.write_log(f'with predictions: {self.best["preds"]}')
        self.write_log(f'of labels:        {self.best["labels"]}')
        self.write_log(self.best["report"])
        self.write_log('')