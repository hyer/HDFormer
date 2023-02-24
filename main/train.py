#!/usr/bin/env python
import os
from functools import reduce
import time
import random
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from torch_optimizer import AdaMod
from torch.optim import Adam, AdamW

from base.baseTrainer import reduce_tensor, save_checkpoint, load_state_dict
from base.utilities import get_parser, get_logger, main_process, AverageMeter, find_free_port
from models import get_model
from metrics import *
from models.base import *


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(args.manual_seed)
    if hasattr(args, 'deterministic') and args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        print("==> Set training to be deterministic!")
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.train_gpu = args.train_gpu[0]
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = args.dist_url + ':%d' % port
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def worker_init_fn(worker_id):
    manual_seed = 12345
    random.seed(manual_seed + worker_id)
    np.random.seed(manual_seed + worker_id)
    torch.manual_seed(manual_seed + worker_id)
    torch.cuda.manual_seed(manual_seed + worker_id)
    torch.cuda.manual_seed_all(manual_seed + worker_id)


def main_worker(gpu, ngpus_per_node, args):
    cfg = args
    best_metric = 1e10
    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url, world_size=cfg.world_size,
                                rank=cfg.rank)
    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(cfg.log_dir)
    if main_process(cfg):
        logger.info(cfg)
    if cfg.distributed:
        cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
        cfg.batch_size_val = int(cfg.batch_size_val / ngpus_per_node)
        cfg.workers = int(cfg.workers / ngpus_per_node)

    # ####################### Data Loader ####################### #
    if cfg.data_name == 'H36M':
        from dataset.Human36M import H36M_Dataset
        train_data = H36M_Dataset(config=cfg, mode='train', logger=logger)
        val_data = H36M_Dataset(config=cfg, mode='test', logger=logger) if cfg.evaluate else None
    elif cfg.data_name == 'MPIINF3DHP':
        from dataset.MPIINF3DHP import MPIINF3DHP_Dataset
        train_data = MPIINF3DHP_Dataset(config=cfg, mode='train', logger=logger)
        val_data = MPIINF3DHP_Dataset(config=cfg, mode='test', logger=logger) if cfg.evaluate else None
    else:
        raise Exception('Dataset not supported yet'.format(cfg.data_name))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data) if cfg.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=cfg.workers, pin_memory=True,
                                               sampler=train_sampler,
                                               worker_init_fn=worker_init_fn)
    if cfg.evaluate:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data) if cfg.distributed else None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.batch_size_val,
                                                 shuffle=False, num_workers=cfg.workers, pin_memory=True,
                                                 drop_last=False,
                                                 worker_init_fn=worker_init_fn, sampler=val_sampler)

    # ####################### Loss ####################### #
    loss_fn = eval(cfg.loss_fn)
    traj_loss_fn = eval(cfg.traj_loss) if hasattr(cfg, 'traj_loss') else None

    # ####################### Model ####################### #
    model = get_model(train_data.skeleton, cfg, logger)
    if cfg.sync_bn:
        logger.info("using DDP synced BN")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if main_process(cfg):
        logger.info("==> creating model ...")
        model.summary(logger, writer)
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
        traj_loss_fn = torch.nn.parallel.DistributedDataParallel(traj_loss_fn.cuda(),
                                                                 device_ids=[gpu]) if traj_loss_fn is not None else None
    else:
        model = model.cuda()
        traj_loss_fn = traj_loss_fn.cuda() if traj_loss_fn is not None else None

    # ####################### Optimizer ####################### #
    optimizer = eval(cfg.optimizer)(model.parameters(), lr=cfg.base_lr,
                                    weight_decay=cfg.weight_decay)
    lr_scheduler = MultiStepLR(optimizer,
                               milestones=cfg.decay_milestones, gamma=cfg.lr_decay_factor)

    # ####################### Weight ####################### #
    if cfg.weight:
        if os.path.isfile(cfg.weight):
            if main_process(cfg):
                logger.info("==> loading weight '{}'".format(cfg.weight))
            checkpoint = torch.load(cfg.weight, map_location=torch.device('cpu'))
            load_state_dict(model, checkpoint['state_dict'])
            if main_process(cfg):
                logger.info("==> loaded weight '{}'".format(cfg.weight))
        else:
            if main_process(cfg):
                logger.info("==> no weight found at '{}'".format(cfg.weight))
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            if main_process(cfg):
                logger.info("=> loading checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
            cfg.start_epoch = checkpoint['epoch']
            load_state_dict(model, checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_metric = checkpoint['best_metric']
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if main_process(cfg):
                logger.info("==> loaded checkpoint '{}' (epoch {})".format(cfg.resume, checkpoint['epoch']))
        else:
            if main_process(cfg):
                logger.info("==> no checkpoint found at '{}'".format(cfg.resume))

    # ####################### Train ####################### #
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
            if cfg.evaluate:
                val_sampler.set_epoch(epoch)
        measures = train(train_loader, model, loss_fn, optimizer, epoch, cfg, traj_loss=traj_loss_fn)
        epoch_log = epoch + 1
        lr_scheduler.step()
        if main_process(cfg):
            logger.info('=' * 5 + 'TRAIN Epoch: {} '.format(epoch_log)
                        + reduce(lambda x, y: x + y, [m.name + ' {:.3f} '.format(m.avg) for m in measures])
                        + '=' * 5)
            for m in measures:
                writer.add_scalar("train/{}".format(m.name), m.avg, epoch_log)

        is_best = False
        if cfg.evaluate and (epoch_log % cfg.eval_freq == 0):
            measures = validate(val_loader, model, loss_fn, cfg, traj_loss=traj_loss_fn)
            if main_process(cfg):
                logger.info('=' * 5 + 'VAL Epoch: {} '.format(epoch_log)
                            + reduce(lambda x, y: x + y, [m.name + ' {:.3f} '.format(m.avg) for m in measures])
                            + '=' * 5)
                for m in measures:
                    writer.add_scalar("val/{}".format(m.name), m.avg, epoch_log)

            # remember best metric and save checkpoint
            is_best = measures[0].avg < best_metric
            best_metric = min(best_metric, measures[0].avg)
        if (epoch_log % cfg.save_freq == 0) and main_process(cfg):
            save_checkpoint(model,
                            other_state={
                                'epoch': epoch_log,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'best_metric': best_metric,
                                'lr_scheduler': lr_scheduler.state_dict()
                            },
                            sav_path=os.path.join(cfg.log_dir, 'model'),
                            is_best=is_best)


# ############################# Core functions ############################# #
def train(train_loader, model, loss_fn, optimizer, epoch, cfg, traj_loss=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    measures_meter = [AverageMeter(name) for name in
                      ["loss", "loss_dir", "loss_bone", "loss_fk", "loss_vertex", "loss_traj"]]

    model.train()
    end = time.time()
    max_iter = cfg.epochs * len(train_loader)
    for i, data in enumerate(train_loader):
        # pdb.set_trace()
        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1
        pose_2d = data['data_2d']
        pose_2d = pose_2d.cuda(non_blocking=True)
        pose_3d = data['data_3d'].cuda(non_blocking=True)
        mean_3d = data['mean_3d'].cuda(non_blocking=True)
        std_3d = data['std_3d'].cuda(non_blocking=True)
        # pdb.set_trace()
        vertex_pre = model(pose_2d, mean_3d, std_3d)
        loss_vertex = loss_fn(vertex_pre, pose_3d)
        if traj_loss is not None:
            loss_traj = traj_loss(vertex_pre, pose_3d)
            loss = loss_vertex + cfg.traj_loss_weight * loss_traj
        else:
            loss = loss_vertex

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        for m in measures_meter:
            if m.name in locals():
                m.update(eval(m.name).item(), pose_2d.shape[0])
        current_lr = optimizer.param_groups[0]['lr']
        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % cfg.print_freq == 0 and main_process(cfg):
            msg = ['Epoch: [{}/{}][{}/{}] '
                   'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                   'Batch: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                   'Remain: {remain_time} '.format(epoch + 1, cfg.epochs, i + 1, len(train_loader),
                                                   batch_time=batch_time, data_time=data_time,
                                                   remain_time=remain_time)
                   ] + [m.name + ' {:.3f} '.format(m.val) for m in measures_meter]
            logger.info(reduce(lambda x, y: x + y, msg))
            for m in measures_meter:
                writer.add_scalar("train_batch/{}".format(m.name), m.val, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)
    return measures_meter


def validate(val_loader, model, loss_fn, cfg, traj_loss=None):
    measures_meter = [AverageMeter(name) for name in
                      ["mpjpe_val", "loss", "loss_dir", "loss_bone", "loss_fk", "loss_vertex", "loss_traj",
                       "bone_sym_err", "bone_err"]]
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # pdb.set_trace()
            pose_2d = data['data_2d']
            pose_2d = pose_2d.cuda(non_blocking=True)
            pose_2d_flip = data['data_2d_flip']
            pose_2d_flip = pose_2d_flip.cuda(non_blocking=True)
            pose_3d = data['data_3d'].cuda(non_blocking=True)
            # pose_3d_flip = data['data_3d_flip'].cuda(non_blocking=True)
            mean_3d = data['mean_3d'].cuda(non_blocking=True)
            std_3d = data['std_3d'].cuda(non_blocking=True)
            # pdb.set_trace()
            vertex_pre = model(pose_2d, mean_3d, std_3d)
            vertex_pre_flip = model(pose_2d_flip, mean_3d, std_3d)
            vertex_pre = avg_flip(vertex_pre, vertex_pre_flip)
            loss_vertex = loss_fn(vertex_pre, pose_3d)
            if traj_loss is not None:
                loss_traj = traj_loss(vertex_pre, pose_3d)
                loss = loss_vertex + cfg.traj_loss_weight * loss_traj
            else:
                loss = loss_vertex
            bone_sym_err = bone_symmetric_error(vertex_pre, skeleton=val_loader.dataset.skeleton)
            mpjpe_val = mpjpe(vertex_pre, pose_3d)

            for m in measures_meter:
                if m.name in locals():
                    if cfg.distributed:
                        m.update(reduce_tensor(eval(m.name), cfg).item(), pose_2d.shape[0])
                    else:
                        m.update(eval(m.name).item(), pose_2d.shape[0])

    return measures_meter


if __name__ == '__main__':
    main()
