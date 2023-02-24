#!/usr/bin/env python
import os
import random
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm

from base.baseTrainer import load_state_dict
from base.utilities import get_parser, get_logger, main_process, AverageMeter, find_free_port
from models import get_model
from metrics.loss import *
from models.base import *
from main.evaluate import h36m_evaluate,MPIINF3DHP_evaluate


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    cudnn.benchmark = True
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
    if hasattr(args, 'deterministic') and args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        print("==> Set inference to be deterministic!")
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.test_gpu)
    if len(args.test_gpu) == 1:
        args.test_gpu = args.test_gpu[0]
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = args.dist_url + ':%d' % port
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.test_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    cfg = args
    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url, world_size=cfg.world_size,
                                rank=cfg.rank)
    global logger
    logger = get_logger()
    if main_process(cfg):
        logger.info(cfg)
    if cfg.distributed:
        cfg.test_batch_size = int(cfg.test_batch_size / ngpus_per_node)
        cfg.batch_size_val = int(cfg.batch_size_val / ngpus_per_node)
        cfg.workers = int(cfg.workers / ngpus_per_node)
    # ####################### Data Loader ####################### #
    if cfg.data_name == 'H36M':
        from dataset.Human36M import H36M_Dataset
        _ = H36M_Dataset(config=cfg, mode='train', logger=logger)
        test_data = H36M_Dataset(config=cfg, mode='test', logger=logger)
    elif cfg.data_name == 'MPIINF3DHP':
        from dataset.MPIINF3DHP import MPIINF3DHP_Dataset
        _ = MPIINF3DHP_Dataset(config=cfg, mode='train', logger=logger)
        test_data = MPIINF3DHP_Dataset(config=cfg, mode='test', logger=logger)
    else:
        raise Exception('Dataset not supported yet'.format(cfg.data_name))
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data) if cfg.distributed else None
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.test_batch_size,
                                              shuffle=False, num_workers=cfg.workers, pin_memory=True,
                                              drop_last=False, sampler=test_sampler)

    # ####################### Model ####################### #
    model = get_model(test_data.skeleton, cfg, logger)
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        torch.cuda.set_device(gpu)
        model = model.cuda()

    # ####################### Loss ####################### #
    loss_fn = eval(cfg.loss_fn)

    # ####################### Weight ####################### #
    if os.path.isfile(cfg.model_path):
        if main_process(cfg):
            logger.info("==> loading weight '{}'".format(cfg.model_path))
        checkpoint = torch.load(cfg.model_path, map_location=torch.device('cpu'))
        load_state_dict(model, checkpoint['state_dict'])
        if main_process(cfg):
            logger.info("==> loaded weight '{}'".format(cfg.model_path))
    else:
        if main_process(cfg):
            logger.info("==> no weight found at '{}'".format(cfg.model_path))

    # ####################### Evaluate ####################### #
    test_mpjpe = test(test_loader, model, loss_fn, cfg)
    logger.info('=' * 5 + 'mpjpe_val: {} '
                .format(test_mpjpe) + '=' * 5
                )


def test(data_loader, model, loss_fn, cfg):
    mpjep_meter = AverageMeter()
    preds_3d, gts_3d, indices = [], [], []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            # pdb.set_trace()
            idx = data['idx']
            pose_2d = data['data_2d']
            pose_2d = pose_2d.cuda(non_blocking=True)
            pose_2d_flip = data['data_2d_flip']
            pose_2d_flip = pose_2d_flip.cuda(non_blocking=True)
            pose_3d = data['data_3d'].cuda(non_blocking=True)
            # pose_3d_flip = data['data_3d_flip'].cuda(non_blocking=True)
            mean_3d = data['mean_3d'].cuda(non_blocking=True)
            std_3d = data['std_3d'].cuda(non_blocking=True)
            vertex_pre = model(pose_2d, mean_3d, std_3d)
            vertex_pre_flip = model(pose_2d_flip, mean_3d, std_3d)
            vertex_pre = avg_flip(vertex_pre, vertex_pre_flip)
            # vertex_pre = avg_flip(vertex_pre, vertex_pre_flip, mean_3d, std_3d)
            mpjep_meter.update(mpjpe(vertex_pre, pose_3d), n=vertex_pre.shape[0])

            preds_3d.append(vertex_pre.permute(0, 3, 1, 2).contiguous().cpu().numpy())
            gts_3d.append(pose_3d.permute(0, 3, 1, 2).contiguous().cpu().numpy())
            indices.append(idx.data.numpy())
    logger.info('=' * 5 + 'mpjpe_val: {} '
                .format(mpjep_meter.avg) + '=' * 5
                )
    preds_3d = np.concatenate(preds_3d, 0)
    gts_3d = np.concatenate(gts_3d, 0)
    indices = np.concatenate(indices, 0)
    if cfg.data_name == 'MPIINF3DHP':
        MPIINF3DHP_evaluate(preds_3d, gts_3d, indices, data_loader.dataset, config=cfg)  # need BVCT
    else:
        os.makedirs(cfg.save_folder, exist_ok=True)
        h36m_evaluate(preds_3d, gts_3d, indices, data_loader.dataset, config=cfg)  # need BVCT
    return mpjep_meter.avg


if __name__ == '__main__':
    main()
