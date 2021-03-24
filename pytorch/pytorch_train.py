import os
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

from pytorch.mri_dataset import MRIDataset
from pytorch.pytorch_resnet import PytorchResNet3D
from augmentation.augment_data import Augmentor

USE_GPU = True


def report(labels, preds):
    if len(set(preds)) > 1:
        return classification_report(labels, preds, target_names=['healthy', 'abnormal'], zero_division=0)
    return 'Only one class predicted'


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

        self.augmentor = Augmentor(args.feature_shape)
        self.train_dataset = MRIDataset(self.train_data_path, True, args.feature_shape)
        self.test_dataset = MRIDataset(self.test_data_path, False, args.feature_shape, preprocess=self.augmentor.process_test_set)

        self.summary = SummaryWriter(self.logdir, f'fold{self.fold}')

        self.write_log(f'Fold: {self.fold}', 0)

        if USE_GPU and torch.cuda.is_available():
            print("USING GPU")
            self.device = torch.device('cuda')
        else:
            print("USING CPU")
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
        self.train_loader = DataLoader(self.train_dataset, self.batch_size, True, collate_fn=self.collate_wrapper_train)#, num_workers=4, persistent_workers=True)
        self.test_loader = DataLoader(self.test_dataset, self.test_size, False)#, num_workers=4, persistent_workers=True)

        # Best Test Results
        self.best = {'iteration': None,
                     'report': None,
                     'preds': None,
                     'labels': None,
                     'MaRIAs': None,
                     'loss': float("inf")}

    def collate_wrapper_train(self, datapairs):
        trans_data = list(zip(*datapairs))

        np_data = [t.numpy() for t in trans_data[0]]
        augmented_data = self.augmentor.augment_batch(np_data)
        axial_data = torch.tensor(augmented_data)
        axial_data = torch.unsqueeze(axial_data, 1)
        return axial_data, torch.stack(trans_data[1], 0)

    # def collate_wrapper_test(self, datapairs):
    #     trans_data = list(zip(*datapairs))
    #
    #     np_data = [t.numpy() for t in trans_data[0]]
    #     augmented_data = self.augmentor.process_test_set(np_data)
    #     print(type(augmented_data))
    #     axial_data = torch.tensor(augmented_data)
    #     axial_data = torch.unsqueeze(axial_data, 1)
    #     return axial_data, torch.stack(trans_data[1], 0)

    def write_log(self, line, train_step):
        self.summary.add_text('Log', line, train_step)

    def log_statistics(self, tag, loss, acc, f1, train_step):
        self.summary.add_scalar('Loss/' + tag, loss, train_step)
        self.summary.add_scalar('Accuracy/' + tag, acc, train_step)
        self.summary.add_scalar('F1 Score/' + tag, f1, train_step)

    def evaluate_on_test(self, network, train_step):

        all_binary_labels, all_preds, all_losses, all_y = [], [], [], []

        network.eval()
        for (x, y) in self.test_loader:

            x = x.to(device=self.device)
            binary_y = torch.where(y == 0, 0, 1).to(device=self.device)

            with torch.no_grad():
                out = network(x)
            preds = out.argmax(dim=1).float()

            loss = F.cross_entropy(out, binary_y)

            all_binary_labels.append(binary_y)
            all_preds.append(preds)
            all_losses += [loss] * len(y)
            all_y.append(y)

        all_binary_labels = torch.cat(all_binary_labels)
        all_preds = torch.cat(all_preds)
        all_losses = torch.stack(all_losses)
        all_y = torch.cat(all_y)

        # Convert back to cpu so can be converted to numpy for statistics
        # TODO: should I just do statistics manually?
        all_preds = all_preds.cpu()
        all_binary_labels = all_binary_labels.cpu()

        test_avg_acc = (all_preds == all_binary_labels).float().mean()
        test_avg_loss = all_losses.mean()
        test_f1 = f1_score(all_binary_labels, all_preds, zero_division=0, average='weighted')
        test_report = report(all_binary_labels, all_preds)

        if test_avg_loss < self.best['loss']:

            self.best['iteration'] = train_step
            self.best['loss'] = test_avg_loss
            self.best['preds'] = all_preds
            self.best['labels'] = all_binary_labels
            self.best['MaRIAs'] = all_y
            self.best['report'] = test_report

            torch.save(network.state_dict(), self.model_save_path)
            print()
            print('===========================> Model saved!')
            print()

        print('Test statistics')
        print('Average Loss:       ', test_avg_loss)
        print('Prediction balance: ', all_preds.mean())
        print(test_report)
        print()

        self.log_statistics('test', test_avg_loss, test_avg_acc, test_f1, train_step)
        self.summary.flush()

    def train(self):

        train_step = 0

        network = PytorchResNet3D(self.feature_shape, self.attention, self.dropout_train_prob)
        network = network.to(device=self.device)
        optimiser = Adam(network.parameters(), lr=self.learning_rate)

        train_accuracies = []
        while train_step <= self.num_batches:

            for (x, y) in self.train_loader:
                network.train()

                x = x.to(device=self.device)
                binary_y = torch.where(y == 0, 0, 1).to(device=self.device)

                out = network(x)
                preds = out.argmax(dim=1).float()

                loss = F.cross_entropy(out, binary_y)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                # Convert back to cpu so can be converted to numpy for statistics
                # TODO: should I just do statistics manually?
                preds = preds.cpu()
                binary_y = binary_y.cpu()

                # Summaries and statistics
                print(f'-- Train Batch {train_step} --')
                print('Loss:               ', loss)
                print('Prediction balance: ', preds.mean())
                print(report(binary_y, preds))
                print()

                train_accuracies.append((preds == binary_y).float().mean())
                running_accuracy = torch.mean(torch.stack(train_accuracies[-self.test_evaluation_period:]))
                train_f1 = f1_score(binary_y, preds, zero_division=0, average='weighted')

                self.summary.add_scalar('Loss/train', loss.item(), train_step)
                self.summary.add_scalar('Accuracy/train', running_accuracy, train_step)
                self.summary.add_scalar('F1 Score/train', train_f1, train_step)
                self.summary.flush()

                if train_step % self.test_evaluation_period == 0:
                    self.evaluate_on_test(network, train_step)

                train_step += 1

        print('Training finished!')
        print(self.best["report"])

        self.write_log(f'Best loss (iteration {self.best["iteration"]}): {self.best["loss"]}', train_step)
        self.write_log(f'with predictions: {self.best["preds"]}', train_step)
        self.write_log(f'of labels:        {self.best["labels"]}', train_step)
        self.write_log(f'with MaRIA scores:{self.best["MaRIAs"]}', train_step)
        self.write_log(self.best["report"], train_step)

        self.summary.close()
