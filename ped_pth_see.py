import pprint
import argparse
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tqdm
import datasets
import models as models
from utils import utils
from config import config, update_config
import core
import numpy as np

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():

    parser = argparse.ArgumentParser(description='ASTNet for Anomaly Detection')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    # update_config(config, args)
    return args

def decode_input(input, train=True):
    video = input['video']
    video_name = input['video_name']
    inputs = video[:-1]
    target = video[-1]
    return inputs, target


def main():
    args = parse_args()
    update_config(config, args)

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()

    gpus = list(config.GPUS)
    model = models.get_net(config)
    logger.info('Model: {}'.format(model.get_name()))
    model=model.cuda()

    #model = nn.DataParallel(model, device_ids=gpus).cuda()
    #model = nn.DataParallel(model, device_ids=gpus).cuda(device=gpus[0])
    logger.info('Epoch: '.format(args.model_file))

    test_dataset = eval('datasets.get_test_data')(config)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        #batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    mat_loader = datasets.get_label(config)
    mat = mat_loader()

    # load model
    state_dict = torch.load(args.model_file)
    #print(state_dict)
    #for key, value in state_dict.items():
        #print(key, value, sep="  ")
        #print(key, value.size(), sep="  ")
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
        #print(state_dict.keys())
    #model.module.load_state_dict(state_dict)


if __name__ == '__main__':
    main()
