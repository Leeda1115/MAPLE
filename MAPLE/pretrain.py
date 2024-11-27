import logging
import os
import time
import random
import json
from tqdm import tqdm
import sys
import torch
import math
from itertools import chain
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
from model.clustering import run_kmeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from configs.opts import parser
from model.main_model_2 import AV_VQVAE_Encoder, AV_VQVAE_Decoder, Cross_PCLEMA
from model.CLUB import CLUBSample_group
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.container import metricsContainer
from utils.Recorder import Recorder
import torch.nn.functional as F
import pickle

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from visualize import visualize_latent_and_proto
from ptflops import get_model_complexity_info

# =================================  seed config ============================
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =============================================================================
def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

def main():
    # utils variable
    global args, logger, dataset_configs
    # statistics variable
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0
    # configs
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()
    # select GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0")

    '''Create snapshot_pred dir for copying code and saving models '''
    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    if os.path.isfile(args.resume):
        args.snapshot_pref = os.path.dirname(args.resume)

    logger = Prepare_logger(args, eval=args.evaluate)

    if not args.evaluate:
        logger.info(f'\nCreating folder: {args.snapshot_pref}')
        logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    else:
        logger.info(f'\nLog file will be save in a {args.snapshot_pref}/Eval.log.')

    '''dataset selection'''
    if args.dataset_name == 'WEAR':
        from dataset.WEAR_dataset import WEARDataset, WEARDataset_untrimmed
    elif args.dataset_name == 'ego4d_cmg':
        from dataset.EGO4D_dataset import EGO4D
    else:
        raise NotImplementedError

    '''Dataloader selection'''
    if args.dataset_name == 'ego4d_cmg':
        feature_path = ''
        labels_pickle = ''
        train_dataloader = DataLoader(
            EGO4D(feature_path, labels_pickle, ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        video_dim = 1536
        audio_dim = 768
        video_output_dim = 512
        audio_output_dim = 256
        n_embeddings = 600
    elif args.dataset_name == 'WEAR':
        data_root = ""
        json_file = ""
        train_dataloader = DataLoader(
            WEARDataset_untrimmed(data_root, json_file, split='Training'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        video_dim = 2048
        audio_dim = 650
        n_embeddings = 400
        video_output_dim = 512
        audio_output_dim = 256
    else:
        raise NotImplementedError

    '''model setting'''

    embedding_dim = 256
    start_epoch = -1
    model_resume = False
    total_step = 0

    Encoder = AV_VQVAE_Encoder(video_dim, audio_dim, video_output_dim, audio_output_dim, n_embeddings, 256)
    Encoder.double()
    Encoder.to(device)
    flops, params = get_model_complexity_info(Encoder, (10, 1536), as_strings=True,
                                              print_per_layer_stat=False)
    print(f"FLOPs: {flops}, Params: {params}")

    Cross_MB = Cross_PCLEMA(n_embeddings, embedding_dim)
    Cross_MB.double()
    Cross_MB.to(device)
    flops, params = get_model_complexity_info(Cross_MB, (10, 256), as_strings=True,
                                              print_per_layer_stat=True)
    print(f"FLOPs: {flops}, Params: {params}")
    Video_mi_net = CLUBSample_group(x_dim=embedding_dim, y_dim=video_output_dim, hidden_size=256)
    Audio_mi_net = CLUBSample_group(x_dim=embedding_dim, y_dim=audio_output_dim, hidden_size=256)

    Decoder = AV_VQVAE_Decoder(video_dim, audio_dim, video_output_dim, audio_output_dim, embedding_dim)

    Video_mi_net.double()
    Audio_mi_net.double()
    Decoder.double()

    '''optimizer setting'''

    Video_mi_net.to(device)
    Audio_mi_net.to(device)
    Decoder.to(device)
    optimizer = torch.optim.Adam(
        chain(Encoder.parameters(), Decoder.parameters()), lr=args.lr)
    optimizer_video_mi_net = torch.optim.Adam(Video_mi_net.parameters(), lr=args.mi_lr)
    optimizer_audio_mi_net = torch.optim.Adam(Audio_mi_net.parameters(), lr=args.mi_lr)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)

    '''loss'''
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    if model_resume is True:
        path_checkpoints = "/home/share2/lida/checkpoint/ego_0.01/your-model-17.pt"
        checkpoints = torch.load(path_checkpoints)
        Encoder.load_state_dict(checkpoints['Encoder_parameters'])
        Video_mi_net.load_state_dict(checkpoints['Video_mi_net_parameters'])
        Audio_mi_net.load_state_dict(checkpoints['Audio_mi_net_parameters'])
        Decoder.load_state_dict(checkpoints['Decoder_parameters'])
        optimizer.load_state_dict(checkpoints['optimizer'])
        optimizer_audio_mi_net.load_state_dict(checkpoints['optimizer_audio_mi_net'])
        optimizer_video_mi_net.load_state_dict(checkpoints['optimizer_video_mi_net'])
        start_epoch = checkpoints['epoch']
        total_step = checkpoints['total_step']
        logger.info("Resume from number {}-th model.".format(start_epoch))

    '''Training and Evaluation'''

    for epoch in range(start_epoch + 1, args.n_epoch):
        loss, total_step = train_epoch(Encoder, Audio_mi_net, Video_mi_net, Decoder, train_dataloader,
                                       video_dim, criterion_event,
                                       optimizer, optimizer_audio_mi_net, optimizer_video_mi_net, epoch, total_step,
                                       args)

        save_path = os.path.join(args.model_save_path, 'your-model-{}.pt'.format(epoch))
        save_models(Encoder, Audio_mi_net, Video_mi_net, Decoder, optimizer, optimizer_audio_mi_net,
                    optimizer_video_mi_net, epoch, total_step, save_path)
        logger.info(f"epoch: ******************************************* {epoch}")
        logger.info(f"loss: {loss}")
        scheduler.step()


def _export_log(epoch, total_step, batch_idx, lr, loss_meter):
    msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, batch_idx, lr)
    for k, v in loss_meter.items():
        msg += '{} = {:.4f}, '.format(k, v)
    logger.info(msg)
    sys.stdout.flush()
    loss_meter.update({"batch": total_step})


def to_eval(all_models):
    for m in all_models:
        m.eval()


def to_train(all_models):
    for m in all_models:
        m.train()


# If resuming training is not required, downstream tasks only need to save the encoder & epoch &  as these are the only components needed for inference.
def save_models(Encoder, Audio_mi_net, Video_mi_net, Decoder, optimizer, optimizer_audio_mi_net,
                optimizer_video_mi_net, epoch_num, total_step, path):
    state_dict = {
        'Encoder_parameters': Encoder.state_dict(),
        'Video_mi_net_parameters': Video_mi_net.state_dict(),
        'Audio_mi_net_parameters': Audio_mi_net.state_dict(),
        'Decoder_parameters': Decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_video_mi_net': optimizer_video_mi_net.state_dict(),
        'optimizer_audio_mi_net': optimizer_audio_mi_net.state_dict(),
        'epoch': epoch_num,
        'total_step': total_step
    }
    torch.save(state_dict, path)
    logging.info('save model to {}'.format(path))


def train_epoch_check(train_dataloader, epoch, total_step, args):
    train_dataloader = tqdm(train_dataloader)
    for n_iter, batch_data in enumerate(train_dataloader):
        '''Feed input to model'''
        visual_feature, audio_feature = batch_data
        visual_feature.cuda()
        audio_feature.cuda()

    return torch.zeros(1)

def train_epoch(Encoder, Audio_mi_net, Video_mi_net, Decoder, train_dataloader, video_dim,
                criterion_event, optimizer, optimize_audio_mi_net, optimizer_video_mi_net, epoch, total_step, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()
    models = [Encoder, Audio_mi_net, Video_mi_net, Decoder]
    to_train(models)

    Encoder.cuda()
    Audio_mi_net.cuda()
    Video_mi_net.cuda()
    Decoder.cuda()
    optimizer.zero_grad()
    mi_iters = 5

    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''

        # ego4d_cmg
        feature, labels, _ = batch_data

        if args.dataset_name == 'ego4d_cmg':
            video_feature = feature[:, :, :video_dim]
            audio_feature = feature[:, :, video_dim:]
        elif args.dataset_name == 'WEAR':
            video_feature = feature[:, :, 650:].to(torch.float64)
            audio_feature = feature[:, :, :650].to(torch.float64)

        # video_feature, audio_feature = batch_data
        video_feature = video_feature.to(torch.float64)
        audio_feature = audio_feature.to(torch.float64)

        audio_feature.cuda()
        video_feature.cuda()

        for i in range(mi_iters):
            optimizer_video_mi_net, lld_video_loss, optimize_audio_mi_net, lld_audio_loss = \
                mi_first_forward(audio_feature, video_feature, Encoder, Audio_mi_net, Video_mi_net,
                                 optimize_audio_mi_net, optimizer_video_mi_net, epoch)

        audio_info_loss, video_info_loss, mi_audio_loss, mi_video_loss, \
        audio_recon_loss, video_recon_loss, audio_class, video_class, cmmd_loss \
            = mi_second_forward(audio_feature, video_feature, Encoder, Audio_mi_net, Video_mi_net,
                                Decoder, epoch)

        loss_items = {
            "audio_recon_loss": audio_recon_loss.item(),
            "lld_audio_loss": lld_audio_loss.item(),
            "audio_info_loss": audio_info_loss.item(),
            "video_recon_loss": video_recon_loss.item(),
            "lld_video_loss": lld_video_loss.item(),
            "video_info_loss": video_info_loss.item(),
            "cmmd_loss": cmmd_loss.item()
        }

        metricsContainer.update("loss", loss_items)
        loss = audio_recon_loss + video_recon_loss +mi_audio_loss + mi_video_loss + \
               cmmd_loss + args.contra_effi * (audio_info_loss + video_info_loss)

        if n_iter % 20 == 0:
            _export_log(epoch=epoch, total_step=total_step + n_iter, batch_idx=n_iter, lr=0.0004,
                        loss_meter=metricsContainer.calculate_average("loss"))

        loss.backward()

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            for model in models:
                total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), video_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

    return losses.avg, n_iter + total_step


def mi_first_forward(audio_feature, video_feature, Encoder, Audio_mi_net, Video_mi_net, optimizer_audio_mi_net,
                     optimizer_video_mi_net, epoch):
    optimizer_video_mi_net.zero_grad()
    optimizer_audio_mi_net.zero_grad()

    video_mu, video_std, video_aggregated_embed, video_semantic_bottom, \
    audio_mu, audio_std, audio_aggregated_embed, audio_semantic_bottom, \
    audio_semantic_result, video_semantic_result, audio_encoder_result, video_encoder_result, \
    cpcl_loss = Encoder(audio_feature, video_feature, epoch)

    video_encoder_result = video_encoder_result.detach()
    audio_encoder_result = audio_encoder_result.detach()

    lld_video_loss = -Video_mi_net.loglikeli(video_semantic_result, video_encoder_result)
    lld_video_loss.backward()
    optimizer_video_mi_net.step()

    lld_audio_loss = -Audio_mi_net.loglikeli(audio_semantic_result, audio_encoder_result)
    lld_audio_loss.backward()
    optimizer_audio_mi_net.step()

    return optimizer_video_mi_net, lld_video_loss, optimizer_audio_mi_net, lld_audio_loss


def mi_second_forward(audio_feature, video_feature, Encoder, Audio_mi_net, Video_mi_net, Decoder, epoch):
    # video_club_feature (batch, length, video_dim)
    video_mu, video_std, video_aggregated_embed, video_semantic_bottom, \
    audio_mu, audio_std, audio_aggregated_embed, audio_semantic_bottom, \
    audio_semantic_result, video_semantic_result, audio_encoder_result, video_encoder_result, \
    cpcl_loss = Encoder(audio_feature, video_feature, epoch)

    seq_len = audio_semantic_bottom.shape[1]

    mi_video_loss = Video_mi_net.mi_est(video_semantic_result, video_encoder_result)
    mi_audio_loss = Audio_mi_net.mi_est(audio_semantic_result, audio_encoder_result)

    audio_info_loss = -0.5 * (1 + 2 * audio_std.log() - audio_mu.pow(2) - audio_std.pow(2)).sum(1).mean().div(
        math.log(2))
    video_info_loss = -0.5 * (1 + 2 * video_std.log() - video_mu.pow(2) - video_std.pow(2)).sum(1).mean().div(
        math.log(2))

    video_recon_loss, audio_recon_loss, video_class, audio_class \
        = Decoder(audio_feature, video_feature, audio_encoder_result, video_encoder_result, audio_semantic_result, video_semantic_result)

    return audio_info_loss, video_info_loss, mi_audio_loss, mi_video_loss, \
           audio_recon_loss, video_recon_loss, audio_class, video_class, cpcl_loss


def alignment_multi_modal_distribute_loss(v_logvar, v_mu, a_logvar, a_mu):
    element1 = a_logvar / v_logvar
    element2 = ((v_mu - a_mu) * (v_mu - a_mu)) / v_logvar
    loss = -torch.sum(torch.log(element1) - element1 - element2 + 1) / 2

    return loss

def VQ_audio_forward(audio_feature, visual_feature, Encoder, optimizer, epoch):
    audio_vq_forward_loss = Encoder.Audio_vq_forward(audio_feature, visual_feature, epoch)
    audio_vq_forward_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return audio_vq_forward_loss, optimizer


def VQ_video_forward(audio_feature, visual_feature, Encoder, optimizer, epoch):
    optimizer.zero_grad()
    video_vq_forard_loss = Encoder.Video_vq_forward(audio_feature, visual_feature, epoch)
    video_vq_forard_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return video_vq_forard_loss, optimizer


def compute_accuracy_supervised(event_scores, labels):
    labels_foreground = labels[:, :, :-1]
    labels_BCE, labels_evn = labels_foreground.max(-1)
    labels_event, _ = labels_evn.max(-1)
    _, event_class = event_scores.max(-1)
    correct = event_class.eq(labels_event)
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())
    return acc


def save_checkpoint(state_dict, top1, task, epoch):
    model_name = f'{args.snapshot_pref}/model_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)


if __name__ == '__main__':
    main()
