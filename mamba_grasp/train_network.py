import argparse
import datetime
import json
import logging
import os
import sys
from tqdm import tqdm
import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
from torchsummary import summary
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow
from timm import scheduler as sh

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(epoch, net, device, train_data, optimizer, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    for x, y, _, _, _ in tqdm(train_data, position=0):
        batch_idx += 1

        xc = x.to(device)
        yc = [yy.to(device) for yy in y]

        lossd = net.compute_loss(xc, yc)


        p_loss = lossd['losses']['p_loss']
        cos_loss = lossd['losses']['cos_loss']
        sin_loss = lossd['losses']['sin_loss']
        width_loss = lossd['losses']['width_loss']

        loss = lossd['loss']

        if batch_idx % 100 == 0:
            logging.info(
                'Epoch: {}, Batch: {}, Loss: {:0.4f}, Q_loss: {:0.4f}, Sin_loss: {:0.4f}, Cos_loss: {:0.4f}, Width_loss: {:0.4f} '.format(
                    epoch, batch_idx, loss.item(), p_loss.item(), cos_loss.item(), sin_loss.item(),
                    width_loss.item()))

        results['loss'] += loss.item()
        for ln, l in lossd['losses'].items():
            if ln not in results['losses']:
                results['losses'][ln] = 0
            results['losses'][ln] += l.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ema.update()

        # Display the images
        if vis:
            imgs = []
            n_img = min(4, x.shape[0])
            for idx in range(n_img):
                imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                    x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in
                                                  lossd['pred'].values()])
            gridshow('Display', imgs,
                     [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0),
                      (0.0, 1.0)] * 2 * n_img,
                     [cv2.COLORMAP_BONE] * 10 * n_img, 10)
            cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results

if __name__ == '__main__':
    run()
