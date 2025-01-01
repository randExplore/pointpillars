from utils.visualization import plot_tensorboard
import argparse
import os
import csv
import numpy as np


def plot_tensorboard_data(args):
    with open(args.lr_path, 'r') as file:
        lr_data = csv.reader(file)
        lr_data = np.array(list(lr_data))[1:, 1:]
        lr_data = lr_data.astype(np.float32)

    with open(args.momentum_path, 'r') as file:
        momentum_data = csv.reader(file)
        momentum_data = np.array(list(momentum_data))[1:, 1:]
        momentum_data = momentum_data.astype(np.float32)

    with open(args.train_loss_path, 'r') as file:
        train_loss_data = csv.reader(file)
        train_loss_data = np.array(list(train_loss_data))[1:, 1:]
        train_loss_data = train_loss_data.astype(np.float32)

    with open(args.val_loss_path, 'r') as file:
        val_loss_data = csv.reader(file)
        val_loss_data = np.array(list(val_loss_data))[1:, 1:]
        val_loss_data = val_loss_data.astype(np.float32)

    save_path = args.save_fig_path
    save_fig = not args.no_save_fig
    plot_tensorboard(lr_data, momentum_data,
                     train_loss_data, val_loss_data,
                     save_path, save_fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration Plot_Tensorboard Parameters")
    parser.add_argument("--lr_path",
                        default=os.path.join(os.getcwd(), "model", "point_pillar",
                                             "summary", "tensorboard_data", "lr.csv"),
                        help="Tensorboard learning rate data path in CSV format")
    parser.add_argument("--momentum_path",
                        default=os.path.join(os.getcwd(), "model", "point_pillar",
                                             "summary", "tensorboard_data", "momentum.csv"),
                        help="Tensorboard momentum data path in CSV format")
    parser.add_argument("--train_loss_path",
                        default=os.path.join(os.getcwd(), "model", "point_pillar",
                                             "summary", "tensorboard_data", "train_loss.csv"),
                        help="Tensorboard training loss data path in CSV format")
    parser.add_argument("--val_loss_path",
                        default=os.path.join(os.getcwd(), "model", "point_pillar",
                                             "summary", "tensorboard_data", "val_loss.csv"),
                        help="Tensorboard validation loss data path in CSV format")
    parser.add_argument("--save_fig_path",
                        default=os.path.join(os.getcwd(), "model", "point_pillar",
                                             "summary", "tensorboard_data"),
                        help="Folder path for the figures to save")
    parser.add_argument('--no_save_fig', action='store_true',
                        help='Select to save plots')

    args = parser.parse_args()
    plot_tensorboard_data(args)
