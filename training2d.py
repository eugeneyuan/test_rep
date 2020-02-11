# -*- coding: utf-8 -*-
import os
import logging
import argparse
# import sys
import time
import shutil
import pandas as pd
import numpy as np

import torch
from torch.nn import DataParallel, SmoothL1Loss
from torch import optim
from torch.utils.data import DataLoader

from src.default_configs.config2d import cfg, _set_log_dir
from src.data.aug3d import Compose, NPRandomFlip, ITKRandomRotateTranslateScale, NPRandomGridDeform
from src.data.dataset import NPYSlice, NPYSliceEval
from src.utils.miscs import log_init
from src.utils.warmup_scheduler import WarmupCosineLR

from src.archs.net2d.simple_unet import UNet

# torch.backends.cudnn.benchmark = True
net_settings = {
    'encoder': [[64, ], ["max", 128, 128], ["max", 256, 256], ["max", 512, 512]],
    'mid': ["max", 512, 512],
    'decoder': [[512, 512], [256, 256], [128, 128], [64, ], ]
}


def slice_test(cfg, net, data_df, score_fn):
    score = 0.
    net.eval()
    device = torch.device(cfg.DEVICE)
    for i in range(len(data_df)):
        file_name = data_df.iloc[i]["file name"]
        length = data_df.iloc[i]["length"]
        data = NPYSliceEval(cfg, file_name, length)
        data_loader = DataLoader(data, batch_size=cfg.VAL.BATCH_SIZE, shuffle=False, num_workers=cfg.DATA.NUM_WORKERS)
        score += test(net, data_loader, score_fn, device)
    return score / len(data_df)


def test(net, data_loader, score_fn, device):
    test_out = []
    test_gt = []
    test_mask = []
    with torch.no_grad():
        for test_iter, (src_imgs, dst_imgs, src_masks) in enumerate(data_loader):
            test_gt.append(dst_imgs.squeeze(dim=1).cpu().numpy())
            test_mask.append(src_masks.squeeze(dim=1).cpu().numpy())

            src_imgs = src_imgs.to(device)

            out = net(src_imgs)

            test_out.append(out.squeeze(dim=1).cpu().numpy())

    test_out = np.concatenate(test_out, axis=0)
    test_gt = np.concatenate(test_gt, axis=0)
    test_mask = np.concatenate(test_mask, axis=0)
    return score_fn(test_out, test_gt, test_mask)


def train(net, data_loader, loss_fn, opt, device, logger=None, iter_count=None, lr_scheduler=None):
    training_loss = 0.
    for train_iter, (src_imgs, dst_imgs, src_masks) in enumerate(data_loader):

        tic = time.time()

        src_imgs = src_imgs.to(device)
        dst_imgs = dst_imgs.to(device)
        src_masks = src_masks.to(device)
        out = net(src_imgs)

        loss = loss_fn(out, dst_imgs, src_masks)

        opt.zero_grad()
        loss.backward()
        opt.step()

        toc = time.time()
        training_loss += loss.item()

        if iter_count is not None:
            iter_count += 1
        if lr_scheduler is not None:
            if iter_count is not None:
                lr_scheduler.step(iter_count)
            else:
                lr_scheduler.step()
        if logger is not None:
            if iter_count is not None:
                logger.info("Iteration %d time %.2fs loss %.4f.", iter_count, toc - tic, loss.item())
            else:
                logger.info("Iteration %d time %.2fs loss %.4f.", train_iter + 1, toc - tic, loss.item())
    if iter_count is not None:
        return training_loss / len(data_loader), iter_count
    else:
        return training_loss / len(data_loader)


def main():
    # command arguments
    parser = argparse.ArgumentParser(description='Processing arguments.')
    parser.add_argument("-c", "--config", type=str, default="./configs/unet2d.yaml", help="config path")
    args = parser.parse_args()
    if args.config != "":
        cfg.merge_from_file(args.config)
    cfg.freeze()

    # augmentations
    augmentations = Compose([
        NPRandomFlip(2, 0.5),
        ITKRandomRotateTranslateScale(theta_x=3, theta_y=3, theta_z=3, tx=10, ty=10, tz=10, scale=0.1, do_probability=0.5),
    ])

    # log init and copy config file and training script
    device = torch.device(cfg.DEVICE)
    save_dir = _set_log_dir(cfg)
    shutil.copy(os.path.basename(__file__), save_dir)
    shutil.copy(args.config, save_dir)
    logger, writer = log_init(save_dir)
    logger.info(cfg)

    # get train/val/test info
    train_df = pd.read_csv(os.path.join(cfg.DATA.DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(cfg.DATA.DATA_DIR, 'val.csv'))
    test_df = pd.read_csv(os.path.join(cfg.DATA.DATA_DIR, 'test.csv'))

    train_data = NPYSlice(cfg, train_df, augmentations=augmentations, phase="train")

    logger.info("Get %d samples for training, %d for validation, %d for testing.",
                len(train_df), len(val_df), len(test_df))

    logger.info("cpu %d", cfg.DATA.NUM_WORKERS)

    train_loader = DataLoader(train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.DATA.NUM_WORKERS)

    # net init
    net = UNet(cfg.DATA.INPUT_SHAPE[0], cfg.NET.NUM_CLASS, net_settings, use_norm=cfg.NET.USE_NORM,
               norm_type="bn", act_type="relu")
    net = net.to(device)

    if cfg.NET.USE_DP:
        net = DataParallel(net)
        opt = optim.Adam(net.module.parameters(), lr=cfg.TRAIN.LR, weight_decay=1e-4)
    else:
        opt = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=1e-4)

    start_epoch = 1

    if cfg.NET.RESUME != "":
        checkpoint = torch.load(cfg.NET.RESUME)
        net.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    lr_scheduler = WarmupCosineLR(opt, max_iters=len(train_loader) * cfg.TRAIN.EPOCHS, warmup_factor=0.001,
                                  warmup_iters=cfg.TRAIN.WARMUP_E * len(train_loader))

    def loss_fn(out_img, dst_img, mask_img, scale=1):
        return SmoothL1Loss(reduction="mean")(scale * out_img[mask_img > 0], scale * dst_img[mask_img > 0])

    def score_fn(out_img, dst_img, mask_img):
        mask = mask_img.astype(np.bool)
        return np.mean(np.abs(out_img[mask] - dst_img[mask])) * 1000

    best_score = np.inf
    best_epoch = 0

    iter_count = 0

    for e in range(start_epoch, cfg.TRAIN.EPOCHS + 1):
        # train
        net.train()
        tic = time.time()
        train_loss, iter_count = train(net, train_loader, loss_fn, opt, device, logger, iter_count, lr_scheduler)
        writer.add_scalar('Loss/train', train_loss, e)
        toc = (time.time() - tic) / 60
        logger.info("Epoch %d takes %.2fm for training, average loss is %.4f.", e, toc, train_loss)

        # save checkpoint
        torch.save({
            'epoch': e,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
        }, os.path.join(save_dir, 'checkpoint.pth'))

        # eval on train set
        net.eval()
        tic = time.time()
        train_score = slice_test(cfg, net, train_df, score_fn)
        toc = (time.time() - tic) / 60
        writer.add_scalar('Score/train', train_score, e)
        logger.info("Epoch %d takes %.2fm evaluating on training set:", e, toc)
        logger.info("Train score %.4f", train_score)

        # eval on val set
        tic = time.time()
        val_score = slice_test(cfg, net, val_df, score_fn)
        toc = (time.time() - tic) / 60
        writer.add_scalar('Score/val', val_score, e)
        logger.info("Epoch %d takes %.2fm evaluating on validation set:", e, toc)
        logger.info("Val score %.4f", val_score)

        # save best epoch
        if val_score <= best_score:
            best_score = val_score
            best_epoch = e
            if cfg.NET.USE_DP:
                torch.save(net.module.state_dict(), os.path.join(save_dir, 'best.pth'))
            else:
                torch.save(net.state_dict(), os.path.join(save_dir, 'best.pth'))

    # save last epoch
    if cfg.NET.USE_DP:
        torch.save(net.module.state_dict(), os.path.join(save_dir, 'last.pth'))
    else:
        torch.save(net.state_dict(), os.path.join(save_dir, 'last.pth'))
    writer.close()
    del train_data, train_loader

    # test
    if cfg.NET.USE_DP:
        net.module.load_state_dict(torch.load(os.path.join(save_dir, 'best.pth')))
    else:
        net.load_state_dict(torch.load(os.path.join(save_dir, 'best.pth')))
    net.eval()
    test_score = slice_test(cfg, net, test_df, score_fn)
    logger.info("Testing using best epoch %d:", best_epoch)
    logger.info("Test score %.4f", test_score)

    logging.shutdown()
    logger.handlers.clear()
    del logger


if __name__ == '__main__':
    main()

