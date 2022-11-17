from __future__ import print_function

import os
import sys
import argparse
import time
import math
import timm
import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from question_loader import Question1Dataset, Question2Dataset, Question3Dataset, Question4Dataset, Group2Dataset
from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from networks.vit import SupConVit
from losses import SupConLoss
import wandb
from calculator import calculate1, calculate3, calculate4, calculate4_both
from quiz_master import quiz3, quiz1, quiz2,  quiz4, group2, group4_question1, group4_question2
from encoder_models.vit import RankVit, ClusterVit
from encoder_models.resnet import Resnet
from tqdm import tqdm
from glob import glob

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

MODELS = {
    'vit': ClusterVit,
    'resnet': Resnet,
    'rankvit': RankVit
}

CALCULATOR = {
    1: calculate1,
    2: calculate1,
    3: calculate3,
    4: calculate4,
    5: calculate1,
    7: calculate1,
    8: calculate4_both,
}

QUIZ_OPTIONS = {
    1: quiz1,
    2: quiz2,
    3: quiz3,
    4: quiz4,
    5: group2,
    7: group4_question1,
    8: group4_question2,
}

def f1_graph(epoch,question_number,model_name,weights_path,opt,question_dir,file_name,title):
    score_model = MODELS[model_name](
            weights_path=weights_path,
            mean=eval(opt.mean),
            std=eval(opt.std),
            question_dir=question_dir,
        )

    score_model.encode_images()

    QUIZ_OPTIONS[question_number](
        score_model,
        question_dir,
        file_name,
    )
    var=CALCULATOR[question_number](
        #answer_file= os.path.join(
        #opt.save_folder,f'ckpt_{opt.method}_pretrained_{opt.pretrained}_{opt.group_num}_{file_name}_epoch_{epoch}.csv'),
        answer_file = file_name,
    )
    print('VAR->',var)
    var = list(var)

    wandb.log({'micro_f1'+title: var[0]}, step=epoch)
    wandb.log({'macro_f1'+title: var[1]}, step=epoch)

def main():
    wandb.init(project="50percent_val_test", entity="newbornking999")

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--mean', type=str,default="(0.6958, 0.6816, 0.6524)")
    parser.add_argument('--root_dir', type=str,default= "/media0/chris/group4_resize_v2")
    parser.add_argument('--std', type=str,default= "(0.3159, 0.3100, 0.3385)")
    parser.add_argument('--weights_path', type=str,default= "/Gits/chris/SupContrast/save/SupCon/path_models/SupCon_path_resnet50_lr_0.005_decay_0.0001_bsz_256_temp_0.1_trial_0_cosine")
    opt = parser.parse_args()

    weights = sorted(glob(f"{weights_path}/*.pth"))
    model_name = 'vit'
    valid_path = "valid"
    test_path = "test"
    #for epoch,weights_path in enumerate(weights):
    for epoch in range(len(weights)):
        epoch  = (epoch +1) * 10
        weights_path_epoch = f"{weights_path}/ckpt_SupCon_pretrained_False_group4_epoch_{epoch}.pth"
        print('weights_path->>',weights_path_epoch)
        file_name = weights_path_epoch.replace('pth','csv')

        question1_val_dir = f"{root_dir}/{valid_path}/question1"
        f1_graph(epoch,7,model_name,weights_path_epoch,opt,question1_val_dir,file_name,"question1_valid")
        
        question1_tst_dir = f"{root_dir}/{test_path}/question1"
        f1_graph(epoch,7,model_name,weights_path_epoch,opt,question1_tst_dir,file_name,"question1_test")

        question2_val_dir = f"{root_dir}/{valid_path}/question2"
        f1_graph(epoch,8,model_name,weights_path_epoch,opt,question2_val_dir,file_name,"question2_valid")

        question2_tst_dir = f"{root_dir}/{test_path}/question2"
        f1_graph(epoch,8,model_name,weights_path_epoch,opt,question2_tst_dir,file_name,"question2_test")
       

if __name__ == '__main__':
    main()