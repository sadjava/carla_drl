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

from carla_drl.semantic_segmentation.dataset import CarlaSemanticSegmentationDataset, labels_to_cityscapes_palette
from carla_drl.semantic_segmentation.unet import UNet
from carla_drl.semantic_segmentation.utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d, \
                                                    DeNormalize
from carla_drl.semantic_segmentation import joint_transforms

cudnn.benchmark = True

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training')
    parser.add_argument('--ckpt_path', default='results/semantic_segmentation', help='path to save checkpoints')
    parser.add_argument('--exp_name', default='im20_160x80', help='experiment name')
    parser.add_argument('--root', default='data_480x270', help='path to the dataset root directory')
    parser.add_argument('--train_batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--epoch_num', type=int, default=500, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--input_size', type=int, nargs=2, default=[160, 80], help='input image size')
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
    model = UNet(num_classes=13).to(device)
    num_parameters = sum(p.numel() for p in model.parameters())
    logging.info("Model created with %d parameters.", num_parameters)

    if len(args.snapshot) == 0:
        curr_epoch = 1
        args.best_record = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    else:
        logging.info("Resuming training from %s", args.snapshot)
        model.load_state_dict(torch.load(os.path.join(args.ckpt_path, args.exp_name, args.snapshot)))
        split_snapshot = args.snapshot.split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        args.best_record = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                               'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                               'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11])}
        
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

    dataset = CarlaSemanticSegmentationDataset(
        root=args.root,
        joint_transform=joint_transform,
        input_transform=input_transform
    )
    logging.info("Dataset created with %d samples.", len(dataset))

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(args.validation_split * dataset_size))
    np.random.seed(args.random_seed)
    np.random.shuffle(indices)
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

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=args.lr_patience, min_lr=1e-10)

    criterion = CrossEntropyLoss2d()

    for epoch in range(curr_epoch, args.epoch_num + 1):
        train(train_loader, model, criterion, optimizer, epoch, args, writer, device)
        val_loss = validate(val_loader, model, criterion, optimizer, epoch, args, restore_transform, visualize, writer, device)
        scheduler.step(val_loss)


def train(train_loader, model, criterion, optimizer, epoch, train_args, writer, device):
    train_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        assert inputs.size()[2:] == labels.size()[1:]
        N = inputs.size(0)
        inputs = Variable(inputs).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        assert outputs.size()[2:] == labels.size()[1:]
        assert outputs.size()[1] == CarlaSemanticSegmentationDataset.NUM_CLASSES

        loss = criterion(outputs, labels) / N
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), N)

        curr_iter += 1
        writer.add_scalar('train_loss', train_loss.avg, curr_iter)

        if (i + 1) % train_args.print_freq == 0:
            logging.info('[epoch %d], [iter %d / %d], [train loss %.5f]',
                epoch, i + 1, len(train_loader), train_loss.avg)


def validate(val_loader, net, criterion, optimizer, epoch, train_args, restore, visualize, writer, device):
    net.eval()

    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []

    for vi, data in enumerate(val_loader):
        inputs, gts = data
        N = inputs.size(0)
        with torch.no_grad():
            inputs = inputs.to(device)
            gts = gts.to(device)
            outputs = net(inputs)
        predictions = outputs.data.max(1)[1].squeeze_(1).cpu().numpy()

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

    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, val_loader.dataset.NUM_CLASSES)

    if mean_iu > train_args.best_record['mean_iu']:
        train_args.best_record['val_loss'] = val_loss.avg
        train_args.best_record['epoch'] = epoch
        train_args.best_record['acc'] = acc
        train_args.best_record['acc_cls'] = acc_cls
        train_args.best_record['mean_iu'] = mean_iu
        train_args.best_record['fwavacc'] = fwavacc
        snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
            epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[1]['lr']
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
            gt_pil = labels_to_cityscapes_palette(data[1])
            predictions_pil = labels_to_cityscapes_palette(data[2])
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
    print('[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc))

    print('best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
        train_args.best_record['val_loss'], train_args.best_record['acc'], train_args.best_record['acc_cls'],
        train_args.best_record['mean_iu'], train_args.best_record['fwavacc'], train_args.best_record['epoch']))

    print('-----------------------------------------------------------------------------------------------------------')

    writer.add_scalar('val_loss', val_loss.avg, epoch)
    writer.add_scalar('acc', acc, epoch)
    writer.add_scalar('acc_cls', acc_cls, epoch)
    writer.add_scalar('mean_iu', mean_iu, epoch)
    writer.add_scalar('fwavacc', fwavacc, epoch)
    writer.add_scalar('lr', optimizer.param_groups[1]['lr'], epoch)

    net.train()
    return val_loss.avg

if __name__ == '__main__':
    args = parse_args()
    main(args)
