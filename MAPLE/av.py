import logging
import os
import time
import random
import json
from tqdm import tqdm
import sys
import torch
from itertools import chain
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import wandb
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
from configs.opts import parser
from model.main_model_2 import AV_VQVAE_Encoder as AV_VQVAE_Encoder
from model.main_model_2 import Semantic_Decoder, Action_Decoder

from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.container import metricsContainer
from utils.Recorder import Recorder
import torch.nn.functional as F
from torch.nn import init
import ptflops

# =================================  seed config ============================
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =============================================================================
def main():
    # utils variable
    global args, logger, writer, dataset_configs
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
    print(args.dataset_name)
    if args.dataset_name == 'ego4d_cmg':
        from dataset.EGO4D_dataset import EGO4D
        feature_path = ''
        labels_pickle = ''
        train_dataloader = DataLoader(
            EGO4D(feature_path, labels_pickle,),
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
        verb_class_num = 106
        noun_class_num = 390
    elif args.dataset_name == 'WEAR':
        data_root = ""
        json_file = ""
        train_dataloader = DataLoader(
            WEARDataset(data_root, json_file, split='Validation'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        video_dim = 2048
        audio_dim = 650
        n_embeddings = 50
        video_output_dim = 512
        audio_output_dim = 256
    else: 
        raise NotImplementedError

    '''model setting'''

    embedding_dim = 256
    model_resume = True
    total_step = 0
    wandb.init(project='cmg', config=args)
    if model_resume is True:
        path_checkpoints = ""
        wandb.log({"path_checkpoints":path_checkpoints}, )
        checkpoints = torch.load(path_checkpoints)
        n_embeddings = checkpoints['Encoder_parameters']['Cross_MB.embedding'].shape[0]
        print(n_embeddings)
    
    # AV
    Encoder = AV_VQVAE_Encoder(video_dim, audio_dim, video_output_dim, audio_output_dim, n_embeddings, embedding_dim, args)
    verb_Decoder = Semantic_Decoder(input_dim=embedding_dim, class_num=verb_class_num)
    noun_Decoder = Semantic_Decoder(input_dim=embedding_dim, class_num=noun_class_num)
    Encoder.float()
    verb_Decoder.float()
    noun_Decoder.float()
    Encoder.to(device)
    verb_Decoder.to(device)
    noun_Decoder.to(device)
    optimizer = torch.optim.Adam(chain(Encoder.parameters(), verb_Decoder.parameters(), noun_Decoder.parameters()), lr=args.lr)

    scheduler = MultiStepLR(optimizer, milestones=[15], gamma=0.5)
    
    '''loss'''
    criterion = nn.BCEWithLogitsLoss()
    criterion_event = nn.CrossEntropyLoss()
    # wandb.watch(Encoder)

    Encoder.load_state_dict(checkpoints['Encoder_parameters'], strict=False)
    start_epoch = 0
    logger.info("Resume from number {}-th model.".format(start_epoch))

    '''Tensorboard and Code backup'''
    recorder = Recorder(args.snapshot_pref, ignore_folder="Exps/")
    recorder.writeopt(args)

    '''Training and Evaluation'''

    for epoch in range(start_epoch+1, args.n_epoch):
        
        loss, total_step = train_epoch(Encoder, verb_Decoder, noun_Decoder, train_dataloader, criterion, criterion_event,
                                       optimizer, epoch, total_step, device, args)
        logger.info(f"epoch: *******************************************{epoch}")

        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            loss = validate_epoch(Encoder, verb_Decoder, noun_Decoder, train_dataloader, criterion, criterion_event, epoch, device)
            logger.info("-----------------------------")
            logger.info(f"evaluate loss:{loss}")
            logger.info("-----------------------------")
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

def save_models(Encoder, optimizer, epoch_num, total_step, path):
    state_dict = {
        'Encoder_parameters': Encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch_num,
        'total_step': total_step,
    }
    torch.save(state_dict, path)
    logging.info('save model to {}'.format(path))


def train_epoch(Encoder, verb_Decoder, noun_Decoder, train_dataloader, criterion, criterion_event, optimizer, epoch, total_step, device, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end_time = time.time()
    models = [Encoder, verb_Decoder, noun_Decoder]
    wandb.watch(models)
    to_train(models)

    Encoder.cuda()
    verb_Decoder.cuda()
    noun_Decoder.cuda()
    # vn_Decoder.cuda()
    optimizer.zero_grad()
    print(epoch)

    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        feature, labels, _ = batch_data
        if args.dataset_name == 'ego4d_cmg':
            video_feature = feature[:, :, :1536]
            audio_feature = feature[:, :, 1536:]
        elif args.dataset_name == 'WEAR':
            video_feature = feature[:, :, 650:]
            audio_feature = feature[:, :, :650]
        video_feature.to(device)
        audio_feature.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        
        if args.v_for_train:
            with torch.no_grad():# Freeze Encoder
                video_vq = Encoder.Video_VQ_Encoder(video_feature)
            verb_class = verb_Decoder(video_vq)
            noun_class = noun_Decoder(video_vq)

            verb_loss = criterion_event(verb_class, labels['verb'])
            noun_loss = criterion_event(noun_class, labels['noun'])
            video_verb_acc = compute_accuracy_supervised(verb_class, labels['verb'])
            video_noun_acc = compute_accuracy_supervised(noun_class, labels['noun'])
            loss_items = {
                "Train/video_verb_loss":verb_loss.item(),
                "Train/video_verb_acc": video_verb_acc.item(),
                "Train/video_noun_loss": noun_loss.item(),
                "Train/video_noun_acc": video_noun_acc.item(),
            }
        else:
            with torch.no_grad():  # Freeze Encoder
                audio_vq = Encoder.Audio_VQ_Encoder(audio_feature)
            verb_class = verb_Decoder(audio_vq)
            noun_class = noun_Decoder(audio_vq)
            verb_loss = criterion_event(verb_class, labels['verb'])
            noun_loss = criterion_event(noun_class, labels['noun'])
            audio_verb_acc = compute_accuracy_supervised(verb_class, labels['verb'])
            audio_noun_acc = compute_accuracy_supervised(noun_class, labels['noun'])
            loss_items = {
                "Train/audio_verb_loss": verb_loss.item(),
                "Train/audio_verb_acc": audio_verb_acc.item(),
                "Train/audio_noun_loss": noun_loss.item(),
                "Train/audio_noun_acc": audio_noun_acc.item(),
            }
        wandb.log(loss_items)
        print(loss_items)
        metricsContainer.update("loss", loss_items)
        loss = verb_loss+noun_loss

        if n_iter % 20 == 0:
            _export_log(epoch=epoch, total_step=total_step+n_iter, batch_idx=n_iter, lr=optimizer.state_dict()['param_groups'][0]['lr'], loss_meter=metricsContainer.calculate_average("loss"))
        loss.backward()


        '''Clip Gradient'''
        if args.clip_gradient is not None:
            for model in models:
                total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), audio_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

    return losses.avg, n_iter + total_step


@torch.no_grad()
def validate_epoch(Encoder, verb_Decoder, noun_Decoder, val_dataloader, criterion, criterion_event, epoch, device,
                   eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    audio_verb_accuracy = AverageMeter()
    audio_noun_accuracy = AverageMeter()
    video_verb_accuracy = AverageMeter()
    video_noun_accuracy = AverageMeter()
    end_time = time.time()

    Encoder.eval()
    verb_Decoder.eval()
    noun_Decoder.eval()
    Encoder.cuda()
    verb_Decoder.cuda()
    noun_Decoder.eval()

    for n_iter, batch_data in enumerate(val_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        feature, labels, _ = batch_data
        if args.dataset_name == 'epic_kitchens_100':
            video_feature = feature[:, :10, :]
            audio_feature = feature[:, 10:, :]
        elif args.dataset_name == 'ego4d_cmg':
            video_feature = feature[:, :, :1536]
            audio_feature = feature[:, :, 1536:]
        video_feature.cuda()
        audio_feature.cuda()
        labels = {k: v.to(device) for k, v in labels.items()}
        bs = video_feature.size(0)

        audio_vq = Encoder.Audio_VQ_Encoder(audio_feature)
        video_vq = Encoder.Video_VQ_Encoder(video_feature)

        video_verb_class = verb_Decoder(video_vq)
        video_noun_class = noun_Decoder(video_vq)
        audio_verb_class = verb_Decoder(audio_vq)
        audio_noun_class = noun_Decoder(audio_vq)

        video_verb_loss = criterion_event(video_verb_class, labels['verb'])
        video_noun_loss = criterion_event(video_noun_class, labels['noun'])
        video_verb_acc = compute_accuracy_supervised(video_verb_class, labels['verb'])
        video_noun_acc = compute_accuracy_supervised(video_noun_class, labels['noun'])

        audio_verb_loss = criterion_event(audio_verb_class, labels['verb'])
        audio_noun_loss = criterion_event(audio_noun_class, labels['noun'])
        audio_verb_acc = compute_accuracy_supervised(audio_verb_class, labels['verb'])
        audio_noun_acc = compute_accuracy_supervised(audio_noun_class, labels['noun'])

        loss = video_verb_loss + video_noun_loss + audio_noun_loss + audio_verb_loss

        audio_verb_accuracy.update(audio_verb_acc.item(), bs)
        audio_noun_accuracy.update(audio_noun_acc.item(), bs)
        video_verb_accuracy.update(video_verb_acc.item(), bs)
        video_noun_accuracy.update(video_noun_acc.item(), bs)
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        losses.update(loss.item(), bs * 10)

    logger.info(
        f'**************************************************************************\t'
        f"\t Audio_verb Evaluation results (acc): {audio_verb_accuracy.avg:.4f}%."
        f"\t Audio_noun Evaluation results (acc): {audio_noun_accuracy.avg:.4f}%."
        f"\t Video_verb Evaluation results (acc): {video_verb_accuracy.avg:.4f}%."
        f"\t Video_noun Evaluation results(acc): {video_noun_accuracy.avg:.4f}%.")
    wandb.log({
        "Val/A_verb Acc": audio_verb_accuracy.avg,
        "Val/A_noun Acc": audio_noun_accuracy.avg,
        "Val/V_verb Acc": video_verb_accuracy.avg,
        "Val/V_noun Acc": video_noun_accuracy.avg,
    })
    return losses.avg


def compute_accuracy_supervised(event_scores, labels):
    _, event_class = event_scores.max(-1)
    # print(event_class, 'labels', labels)
    correct = event_class.eq(labels)
    correct_num = correct.sum()
    acc = correct_num * (100. / correct.numel())
    return acc


if __name__ == '__main__':
    main()
