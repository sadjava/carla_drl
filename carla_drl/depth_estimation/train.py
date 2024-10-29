import datetime
import os
import random
import argparse
import logging

import numpy as np
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, SubsetRandomSampler

from carla_drl.depth_estimation.dataset import CarlaDepthEstimationDataset
from carla_drl.depth_estimation.midas import MonoDepthNet, resize_image, resize_depth
from carla_drl.depth_estimation.utils import AverageMeter, ScaleAndShiftInvariantLoss, evaluate, \
                                            check_mkdir, DeNormalize
from carla_drl.semantic_segmentation import joint_transforms

cudnn.benchmark = True

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Depth Estimation Training with MiDaS')
    parser.add_argument('--ckpt_path', default='results/depth_estimation', help='path to save checkpoints')
    parser.add_argument('--exp_name', default='im20_480x270', help='experiment name')
    parser.add_argument('--root', default='data_480x270', help='path to the dataset root directory')
    parser.add_argument('--train_batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--epoch_num', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--input_size', type=int, nargs=2, default=[480, 270], help='input image size')
    parser.add_argument('--momentum', type=float, default=0.95, help='momentum')
    parser.add_argument('--lr_patience', type=int, default=10, help='learning rate patience')
    parser.add_argument('--snapshot', default='', help='path to pretrained model snapshot')
    parser.add_argument('--print_freq', type=int, default=20, help='print frequency')
    parser.add_argument('--val_batch_size', type=int, default=16, help='validation batch size')
    parser.add_argument('--val_save_to_img_file', action='store_true', help='save validation images')
    parser.add_argument('--val_img_sample_rate', type=float, default=0.05, help='validation image sample rate')
    parser.add_argument('--validation_split', type=float, default=0.2, help='validation split')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    return parser.parse_args()

def main(args: argparse.Namespace):
    setup_logging()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MonoDepthNet().to(device)
    num_parameters = sum(p.numel() for p in model.parameters())
    logging.info("Model created with %d parameters.", num_parameters)

    if len(args.snapshot) == 0:
        curr_epoch = 1
        args.best_record = {'epoch': 0, 'val_loss': 1e10, 'mae': 1e10, 'rmse': 1e10, 'relative_error': 1e10}
    else:
        logging.info("Resuming training from %s", args.snapshot)
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, args.exp_name, args.snapshot)))
        split_snapshot = args.snapshot.split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        args.best_record = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                               'mae': float(split_snapshot[5]), 'rmse': float(split_snapshot[7]),
                               'relative_error': float(split_snapshot[9])}
    
    log_dir = os.path.join(args.ckpt_path, args.exp_name)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    writer = SummaryWriter(os.path.join(log_dir, timestamp))

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    joint_transform = joint_transforms.Compose([
        joint_transforms.Resize(args.input_size),
        joint_transforms.RandomHorizontalFlip()
    ])
    input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=mean, std=std)
    ])
    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=mean, std=std),
        standard_transforms.ToPILImage()
    ])
    visualize = standard_transforms.ToTensor()

    dataset = CarlaDepthEstimationDataset(
        root=args.root,
        joint_transform=joint_transform,
        input_transform=input_transform
    )
    logging.info("Dataset created with %d samples.", len(dataset))

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(args.random_seed)
    np.random.shuffle(indices)
    split = int(np.floor(args.validation_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=args.train_batch_size, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(dataset, batch_size=args.val_batch_size, sampler=val_sampler, num_workers=4)

    logging.info("Training set: %d samples, Validation set: %d samples.", len(train_indices), len(val_indices))

    optimizer = optim.SGD([
        {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args.lr},
        {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'],
         'lr': args.lr, 'weight_decay': args.weight_decay}
    ], momentum=args.momentum)

    if len(args.snapshot) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(args.ckpt_path, args.exp_name, 'opt_' + args.snapshot)))
        optimizer.param_groups[0]['lr'] = 2 * args.lr
        optimizer.param_groups[1]['lr'] = args.lr

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.lr_patience, min_lr=1e-10)
    criterion = torch.nn.MSELoss()

    for epoch in range(curr_epoch, args.epoch_num + 1):
        train(train_loader, model, criterion, optimizer, epoch, args, writer, device)
        val_loss = validate(val_loader, model, criterion, optimizer, epoch, args, restore_transform, visualize, writer, device)
        scheduler.step(val_loss)

def train(train_loader, model, criterion, optimizer, epoch, args, writer, device):
    model.train()
    train_loss = AverageMeter()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        
        N, _, H, W = inputs.shape
        resized_inputs = resize_image(inputs).float().to(device)
        labels = labels.squeeze(1).float().to(device)

        optimizer.zero_grad()
        outputs = resize_depth(model(resized_inputs), H, W).squeeze(1)

        loss = criterion(outputs, labels) / N
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))
        writer.add_scalar('train_loss', train_loss.avg, epoch)

        if (i + 1) % args.print_freq == 0:
            logging.info('[epoch %d], [iter %d / %d], [train loss %.5f]',
                         epoch, i + 1, len(train_loader), train_loss.avg)

def validate(val_loader, net, criterion, optimizer, epoch, train_args, restore, visualize, writer, device):
    net.eval()

    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []

    for vi, data in enumerate(val_loader):
        inputs, gts = data
        N, _, H, W = inputs.shape
        with torch.no_grad():
            resized_inputs = resize_image(inputs).float().to(device)
            gts = gts.squeeze(1).float().to(device)
            outputs = resize_depth(net(resized_inputs), H, W).squeeze(1)
        predictions = outputs.data.cpu().numpy()
        val_loss.update(criterion(outputs, gts).item() / N, N)

        for i in inputs:
            if random.random() > train_args.val_img_sample_rate:
                inputs_all.append(None)
            else:
                inputs_all.append(i.data.cpu())
        gts_all.append(gts.data.cpu().numpy())
        predictions_all.append(predictions)

    gts_all = np.concatenate(gts_all)
    predictions_all = np.concatenate(predictions_all)
    mae, rmse, relative_error = evaluate(predictions_all, gts_all)

    if mae < train_args.best_record['mae']:
        train_args.best_record['val_loss'] = val_loss.avg
        train_args.best_record['epoch'] = epoch
        train_args.best_record['mae'] = mae
        train_args.best_record['rmse'] = rmse
        train_args.best_record['relative_error'] = relative_error
        snapshot_name = 'epoch_%d_loss_%.5f_mae_%.5f_rmse_%.5f_relative-error_%.5f_lr_%.10f' % (
            epoch, val_loss.avg, mae, rmse, relative_error, optimizer.param_groups[1]['lr']
        )
        torch.save(net.state_dict(), os.path.join(train_args.ckpt_path, train_args.exp_name, snapshot_name + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(train_args.ckpt_path, train_args.exp_name, 'opt_' + snapshot_name + '.pth'))

        if train_args.val_save_to_img_file:
            to_save_dir = os.path.join(train_args.ckpt_path, train_args.exp_name, str(epoch))
            check_mkdir(to_save_dir)

        val_visual = []
        for idx, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
            if data[0] is None:
                continue
            input_pil = restore(data[0])
            gt_pil = restore(torch.from_numpy(data[1]))
            predictions_pil = restore(torch.from_numpy(data[2]))
            if train_args.val_save_to_img_file:
                input_pil.save(os.path.join(to_save_dir, '%d_input.png' % idx))
                predictions_pil.save(os.path.join(to_save_dir, '%d_prediction.png' % idx))
                gt_pil.save(os.path.join(to_save_dir, '%d_gt.png' % idx))
            val_visual.extend([visualize(input_pil.convert('RGB')), visualize(gt_pil.convert('RGB')),
                               visualize(predictions_pil.convert('RGB'))])
        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
        writer.add_image(snapshot_name, val_visual)

        print('-----------------------------------------------------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [mae %.5f], [rmse %.5f], [relative_error %.5f]' % (
        epoch, val_loss.avg, mae, rmse, relative_error))

    print('best record: [val loss %.5f], [mae %.5f], [rmse %.5f], [relative_error %.5f], [epoch %d]' % (
        train_args.best_record['val_loss'], train_args.best_record['mae'],
        train_args.best_record['rmse'], train_args.best_record['relative_error'], train_args.best_record['epoch']))

    print('-----------------------------------------------------------------------------------------------------------')

    writer.add_scalar('val_loss', val_loss.avg, epoch)
    writer.add_scalar('mae', mae, epoch)
    writer.add_scalar('rmse', rmse, epoch)
    writer.add_scalar('relative_error', relative_error, epoch)
    writer.add_scalar('lr', optimizer.param_groups[1]['lr'], epoch)

    net.train()
    return val_loss.avg


if __name__ == '__main__':
    args = parse_args()
    main(args)
