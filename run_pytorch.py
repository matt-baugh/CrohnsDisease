import argparse
from pytorch.pytorch_train import PytorchTrainer
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("crohns_or_polyps", help="Task to run", choices=['Crohns_MRI','Polyps_CT'])
    parser.add_argument("base", help="Path to project base")
    parser.add_argument("train_datapath", help="Path to train TF Record")
    parser.add_argument("test_datapath", help="Path to test TF Record")
    parser.add_argument("-record_shape", help="Dimensions of a single dataset feature")
    parser.add_argument("-feature_shape", help="Desired dimensions of input feature to network")
    parser.add_argument("-gpus", help="Choose which GPU's to use")

    # Optional arguments
    parser.add_argument("-py", "--pytorch", help="Use Pytorch model", type=bool, default=False)
    parser.add_argument("-f", "--fold", help="Fold id", default='')
    parser.add_argument("-bS", "--batch_size", help="Batch size", type=int, default=64)
    parser.add_argument("-lD", "--logdir", help="Directory to log Tensorboard to", default='logdir')
    parser.add_argument("-nB", "--num_batches", help="Number of total training batches", type=int, default=None)
    parser.add_argument("-at", "--attention", help="Inclusion of attention layers in network", default=0)
    parser.add_argument("-mode", "--mode", help="Training or testing mode", default="test")
    parser.add_argument("-mP", "--model_path", help="Path to model save", default="CrohnsDisease/trained_models/crohns_model")
    parser.add_argument("-axt2", "--axial_t2", help="Use Axial T2 scans as input for model", type=int, default=1)
    parser.add_argument("-cort2", "--coronal_t2", help="Use Axial T2 scans as input for model", type=int, default=0)
    parser.add_argument("-axpc", "--axial_pc", help="Use Axial Post Contrast scans as input for model", type=int, default=0)
    # parser.add_argument("-ma", "--mixedattention", help="Inclusion of mixed hard-soft attention loss", default=0)
    # parser.add_argument("-lc", "--localisation", help="Terminal Ileum localisation task", default=0)
    # parser.add_argument("-de", "--deeper", help="Depth of network", default=0)

    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')

    # Parse arguments
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    # Parse the arguments
    args = parseArguments()

    # Raw print arguments
    print("Running with arguments: ")
    for a in args.__dict__:
        print(str(a) + ": " + str(args.__dict__[a]))
    args.attention = int(args.attention)
    # args.localisation = int(args.localisation)
    # args.mixedAttention = int(args.mixedAttention)
    # args.deeper = int(args.deeper)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    args.feature_shape = tuple([int(x) for x in args.feature_shape.split(',')])
    args.record_shape = tuple([int(x) for x in args.record_shape.split(',')])

    task = args.__dict__['crohns_or_polyps']

    if not args.pytorch:
        print("This only works with pytorch!!!")

    trainer = PytorchTrainer(args)
    trainer.train()

