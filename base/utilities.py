#!/usr/bin/env python
import argparse
import os
import logging
import socket

from base import config

def get_parser():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--config', type=str, default='**.yaml', help='config file')
    parser.add_argument('opts', help=' ', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    # cfg = Config.fromfile(args.config)
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        # pdb.set_trace()
        # cfg = cfg.merge_from_dict(args.opts)
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d]==>%(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name=None):
        self.reset()
        if name is not None:
            self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def main_process(args):
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port
