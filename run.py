import argparse
import tensorflow as tf

from runner import Runner
from model.vgg import VGG
from model.resnet import ResNet3D
from main_util import generate_decode_function

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Positional mandatory arguments
    parser.add_argument("crohns_or_polyps", help="Task to run", choices=['Crohns_MRI','Polyps_CT'])
    parser.add_argument("train_datapath", help="Path to train TF Record")
    parser.add_argument("test_datapath", help="Path to test TF Record")
    parser.add_argument("feature_shape", help="Dimensions of a single dataset feature")

    # Optional arguments
    parser.add_argument("-bS", "--batch_size", help="Batch size", type=int, default=64)
    parser.add_argument("-lD", "--logdir", help="Directory to log Tensorboard to", default='logdir')

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

    args.__dict__['feature_shape'] = tuple([int(x) for x in args.__dict__['feature_shape'].split(',')])

    task = args.__dict__['crohns_or_polyps']
    if task == 'Polyps_CT':
        decode_record = generate_decode_function(args.feature_shape, 'image')
        model = VGG
    elif task == 'Crohns_MRI':
        decode_record = generate_decode_function(args.feature_shape, 'axial_t2')
        model = ResNet3D
    args.__dict__['decode_record'] = decode_record

    runner = Runner(args, model)
    runner.train()
