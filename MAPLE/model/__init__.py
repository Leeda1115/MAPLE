import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from tqdm import tqdm
import numpy as np
# from preprocess import mulaw_decode
import math
from torch.nn import MultiheadAttention
from model.models import EncoderLayer, Encoder, DecoderLayer
from torch import Tensor
# The model is testing
from model.mine import MINE
from info_nce import InfoNCE
from torch.nn import init as init
import random
from sklearn.neighbors import kneighbors_graph
from numbers import Number

random.seed(123)
import scipy


def sinkhorn_distance_batch(x, y, epsilon=0.1, num_iters=100):
    """
    Compute Sinkhorn distance between batches of empirical distributions with regularization.

    Args:
        x: Tensor of shape [B, T, D], the first batch of distributions
        y: Tensor of shape [B, T, D], the second batch of distributions
        epsilon: Regularization parameter
        num_iters: Number of iterations for Sinkhorn algorithm

    Returns:
        Sinkhorn distances for each batch (Tensor of shape [B])
    """
    B, T, D = x.shape

    # Compute cost matrix
    cost_matrix = torch.cdist(x, y, p=2.0)  # Shape [B, T, T]

    # Initialize u and v
    mu = torch.ones(T, device=x.device) / T  # Shape [T]
    nu = torch.ones(T, device=y.device) / T  # Shape [T]
    u = torch.ones((B, T), device=x.device).double()  # Shape [B, T]
    v = torch.ones((B, T), device=y.device).double()  # Shape [B, T]

    K = torch.exp(-cost_matrix / epsilon).double()  # Shape [B, T, T]

    for _ in range(num_iters):
        u = mu / (torch.matmul(K, v.unsqueeze(-1)).squeeze(-1))
        v = nu / (torch.matmul(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1))

    # Compute the Sinkhorn distance
    transport_plan = u.unsqueeze(-1) * K * v.unsqueeze(1)
    sinkhorn_distance = torch.sum(transport_plan * cost_matrix, dim=[1, 2]).mean()

    return sinkhorn_distance


def wasserstein_distance(mu1, sigma1, mu2, sigma2):
    """
    计算两个高斯分布N(mu1, sigma1)和N(mu2, sigma2)之间的2-Wasserstein距离
    参数:
    mu1 (Tensor): 均值1, 形状为 [D]
    sigma1 (Tensor): 协方差矩阵1, 形状为 [D, D]
    mu2 (Tensor): 均值2, 形状为 [D]
    sigma2 (Tensor): 协方差矩阵2, 形状为 [D, D]
    """

    epsilon = 1e-6
    sigma1 = sigma1 + epsilon * torch.eye(sigma1.shape[-1], device=sigma1.device)

    mu_diff = mu1 - mu2
    mu_dist = (mu_diff * mu_diff).sum(-1)

    sigma1_sqrt = torch.linalg.cholesky(sigma1)
    # print(mu1.shape, sigma1.shape)

    term2 = torch.mm(torch.mm(sigma1_sqrt, sigma2.transpose(-2, -1)), sigma1_sqrt.transpose(-2, -1))
    term2 = term2 + epsilon * torch.eye(term2.shape[-1], device=term2.device)
    term2 = torch.linalg.cholesky(term2)
    sigma_trace = (sigma1 + sigma2).diagonal(dim1=-2, dim2=-1).sum(-1) - \
                  2 * term2.diagonal(dim1=-2, dim2=-1).sum(-1)

    dist = mu_dist + sigma_trace

    return dist


def wasserstein_contra(mu1, sigma1, mu2, sigma2, margin=1.0):
    """
    计算批量样本之间的Wasserstein对比损失
    参数:
    mu1 (Tensor): 样本1的均值, 形状为 [bs, D]
    sigma1 (Tensor): 样本1的协方差矩阵, 形状为 [bs, D, D]
    mu2 (Tensor): 样本2的均值, 形状为 [bs, D]
    sigma2 (Tensor): 样本2的协方差矩阵, 形状为 [bs, D, D]
    margin (float): 对比损失中的边界值
    """

    bs, D = mu1.shape

    epsilon = 1e-6
    sigma1 = sigma1 + epsilon * torch.eye(sigma1.shape[-1], device=sigma1.device)

    mu_diff = mu1.unsqueeze(1) - mu2.unsqueeze(0)  # [bs, bs, D]
    mu_dist = (mu_diff * mu_diff).sum(-1)  # [bs, bs]

    sigma1_sqrt = torch.linalg.cholesky(sigma1)  # [bs, D, D]

    term2 = torch.einsum('bik,sjk->bsij', sigma1_sqrt, sigma2)  # [bs, bs, D, D]
    term2 = torch.einsum('bik,sjk->bsij', term2, sigma1_sqrt)  # [bs, bs, D, D]
    term2 = term2 + 1e-6 * torch.eye(D, device=sigma1.device)  # [bs, bs, D, D]
    term2 = torch.linalg.cholesky(term2)  # [bs, bs, D, D]

    sigma_trace = (sigma1.unsqueeze(1) + sigma2.unsqueeze(0)).diagonal(dim1=-2, dim2=-1).sum(-1) - \
                  2 * term2.diagonal(dim1=-2, dim2=-1).sum(-1)  # [bs, bs]

    dist_matrix = mu_dist + sigma_trace  # [bs, bs]

    positive_pairs = torch.diag(dist_matrix)
    negative_pairs = dist_matrix + margin - 2 * torch.eye(bs, device=mu1.device) * margin

    loss = torch.clamp(positive_pairs.unsqueeze(1) - negative_pairs, min=0)
    loss = loss.sum() / bs

    return loss


# def wasserstein_contra(mu1, sigma1, mu2, sigma2, margin=1.0):
#     bs, D = mu1.shape
#
#     distance_matrix = torch.zeros(bs, bs, device=mu1.device)
#     for i in range(bs):
#         for j in range(bs):
#             distance_matrix[i, j] = wasserstein_distance(mu1[i, :], sigma1[i, :, :], mu2[j, :], sigma2[j, :, :])
#
#     # 计算对比损失
#     positive_pairs = torch.diag(distance_matrix)
#     negative_pairs = distance_matrix + margin - 2 * torch.eye(bs, device=mu1.device) * margin
#
#     loss = torch.clamp(positive_pairs - negative_pairs, min=0)
#     loss = loss.sum() / bs
#
#     return loss


class Mask_MHCrossAttention(nn.Module):
    def __init__(self, video_input_dim, audio_input_dim, latent_dim, num_heads, head_dim):
        super(Mask_MHCrossAttention, self).__init__()
        assert video_input_dim % num_heads == 0, "Input dimension must be divisible by the number of heads"
        assert audio_input_dim % num_heads == 0, "Input dimension must be divisible by the number of heads"

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.query_dim = latent_dim // num_heads
        self.video_input_dim = video_input_dim
        self.audio_input_dim = audio_input_dim

        self.query_projection = nn.Linear(latent_dim, num_heads * self.query_dim)
        self.key_projection = nn.Linear(input_dim, num_heads * self.query_dim)
        self.value_projection = nn.Linear(input_dim, num_heads * self.head_dim)
        self.audio_key_projection = nn.Linear(audio_input_dim, num_heads * self.query_dim)
        self.audio_value_projection = nn.Linear(audio_input_dim, num_heads * self.head_dim)

        self.output_projection = nn.Linear(num_heads * self.head_dim, latent_dim)

    def forward(self, input, latent, attention_mask=None):
        batch_size, seq_length, D = input.size()

        # Project input and latent into queries, keys, and values for all heads
        projected_query = self.query_projection(latent).view(batch_size, seq_length, self.num_heads, self.query_dim)
        if D == self.video_input_dim:
            projected_key = self.key_projection(input).view(batch_size, seq_length, self.num_heads, self.query_dim)
            projected_value = self.value_projection(input).view(batch_size, seq_length, self.num_heads, self.head_dim)
        else:
            projected_key = self.audio_key_projection(input).view(batch_size, seq_length, self.num_heads,
                                                                  self.query_dim)
            projected_value = self.audio_value_projection(input).view(batch_size, seq_length, self.num_heads,
                                                                      self.head_dim)

        # Transpose and reshape for scaled dot-product attention
        projected_query = projected_query.transpose(1, 2).contiguous().view(batch_size * self.num_heads, seq_length,
                                                                            self.query_dim)
        projected_key = projected_key.transpose(1, 2).contiguous().view(batch_size * self.num_heads, seq_length,
                                                                        self.query_dim)
        projected_value = projected_value.transpose(1, 2).contiguous().view(batch_size * self.num_heads, seq_length,
                                                                            self.head_dim)

        # Compute scaled dot-product attention
        attention_scores = torch.bmm(projected_query, projected_key.transpose(1, 2)) / math.sqrt(self.query_dim)
        print(attention_scores.shape)
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        print(attention_scores.shape)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.bmm(attention_weights, projected_value)

        # Reshape and transpose attention output
        attention_output = attention_output.view(batch_size, self.num_heads, seq_length, self.head_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length,
                                                                              self.num_heads * self.head_dim)

        # Project attention output
        output = self.output_projection(attention_output)
        print(output.shape)
        return output


class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4)
        self.encoder = Encoder(self.encoder_layer, num_layers=2)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        # feature = feature+position_emb
        feature = self.encoder(feature)
        return feature


class VIB(nn.Module):
    # Variational Information Bottleneck
    # vib = VIB(input_feature_size, embeding_fea_size)
    def __init__(self, hidden_dim):
        super(self.__class__, self).__init__()
        self.encoder_mu = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_std = nn.Linear(hidden_dim, hidden_dim)

        init.kaiming_normal_(self.encoder_mu.weight, mode='fan_out')
        init.constant_(self.encoder_mu.bias, 0)
        init.kaiming_normal_(self.encoder_std.weight, mode='fan_out')
        init.constant_(self.encoder_std.bias, 0)

    def forward(self, x, num_sample=9):
        mu = self.encoder_mu(x)
        std = F.softplus(self.encoder_std(x) - 5, beta=1)
        encoding = self.reparametrize_n(mu, std, num_sample)
        if num_sample != 1:
            encoding = torch.mean(encoding, dim=0, keepdim=False)
        return (mu, std), encoding

    def reparametrize_n(self, mu, std, n=1):
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1:
            mu = expand(mu)
            std = expand(std)

        eps = std.data.new(std.size()).normal_()
        return mu + eps * std


class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)] * n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


class Video_Semantic_Encoder(nn.Module):
    def __init__(self, video_dim):
        super(Video_Semantic_Encoder, self).__init__()
        self.reduction = 8
        self.aver_pool = nn.AdaptiveAvgPool2d(1)
        self.se_layer = nn.Sequential(
            nn.Linear(video_dim, video_dim // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(video_dim // self.reduction, video_dim, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
        self.affine_video_ave = nn.Linear(video_dim, video_dim // 2)
        self.affine_video_self = nn.Linear(video_dim, video_dim // 2)
        self.ave_v_att = nn.Linear(video_dim // 2, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, video_feat):
        batch, length, h, w, v_dim = video_feat.size()
        video_feat = video_feat.reshape(batch * length, h, w, v_dim)
        average_video_feat = video_feat.permute(0, 3, 1, 2)
        average_video_feat = self.aver_pool(average_video_feat).view(batch * length, v_dim)
        average_attention = self.se_layer(average_video_feat).view(batch * length, v_dim, 1, 1).permute(0, 2, 3, 1)
        video_channel_att = video_feat * average_attention.expand_as(video_feat) + video_feat

        video_average = self.relu(self.affine_video_ave(average_video_feat)).unsqueeze(-2)
        self_video_att_feat = video_channel_att.reshape((batch * length, -1, v_dim))
        self_video_att_query = self.relu(self.affine_video_self(self_video_att_feat))
        self_query = self_video_att_query * video_average
        self_spatial_att_maps = self.softmax(self.tanh(self.ave_v_att(self_query))).transpose(2, 1)

        self_att_feat = torch.bmm(self_spatial_att_maps,
                                  video_channel_att.view(batch * length, -1, v_dim)).squeeze().reshape(batch, length,
                                                                                                       v_dim)

        return self_att_feat


""" class_num AVVP:25+1(negative label) AVE_AVVP:12+1 """


class Semantic_Decoder_AVVP(nn.Module):
    def __init__(self, input_dim, class_num):
        super(Semantic_Decoder_AVVP, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.event_classifier = nn.Linear(input_dim, class_num)

    def forward(self, input_vq):
        class_logits = self.event_classifier(input_vq)
        return class_logits


class Video_Encoder(nn.Module):
    def __init__(self, video_dim, hidden_dim):
        super(Video_Encoder, self).__init__()
        self.video_dim = video_dim
        self.hidden_dim = hidden_dim
        self.video_linear = nn.Linear(video_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, video_feat):
        return self.relu(self.video_linear(video_feat))


class Audio_Encoder(nn.Module):
    def __init__(self, audio_dim, hidden_dim):
        super(Audio_Encoder, self).__init__()
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.audio_linear = nn.Linear(audio_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, audio_feat):
        return self.relu(self.audio_linear(audio_feat))


class Video_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, vq_dim):
        super(Video_Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.relu = nn.ReLU()
        self.video_rec = nn.Linear(input_dim * 2, output_dim)
        self.video_linear = nn.Linear(vq_dim, input_dim)

    def forward(self, video_encoder_result, video_vq):
        video_vq_result = self.video_linear(video_vq)
        video_encoder_result = torch.cat([video_vq_result, video_encoder_result], dim=2)
        video_decoder_result = self.video_rec(video_encoder_result)
        return video_decoder_result


class Audio_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, vq_dim):
        super(Audio_Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.relu = nn.ReLU()
        self.audio_rec = nn.Linear(input_dim * 2, output_dim)
        self.audio_linear = nn.Linear(vq_dim, input_dim)

    def forward(self, audio_encoder_result, audio_vq):
        audio_vq_result = self.audio_linear(audio_vq)
        audio_encoder_result = torch.cat([audio_vq_result, audio_encoder_result], dim=2)
        audio_decoder_result = self.audio_rec(audio_encoder_result)
        return audio_decoder_result


""" class_num AVE:28  VGGSOUND:141  UCF_VGGSOUND:16 """


class Semantic_Decoder(nn.Module):
    def __init__(self, input_dim, class_num):
        super(Semantic_Decoder, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.event_classifier = nn.Linear(input_dim, class_num)

    def forward(self, input_feat):
        input_feat = self.linear(input_feat)
        input_feat, _ = input_feat.max(1)
        # input_feat = input_feat + cls_token
        class_logits = self.event_classifier(input_feat)
        return class_logits


class Action_Decoder(nn.Module):
    def __init__(self, hidden_dim, verb_class_num, noun_class_num):
        super(Action_Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.verb_cls = nn.Parameter(torch.randn(1, self.hidden_dim))
        self.noun_cls = nn.Parameter(torch.randn(1, self.hidden_dim))

        self.multihead_attn = nn.MultiheadAttention(hidden_dim, 8)

        self.verb_classifier = nn.Linear(hidden_dim, verb_class_num)
        self.noun_classifier = nn.Linear(hidden_dim, noun_class_num)

    def forward(self, semantic_feat):
        bs = semantic_feat.shape[0]
        verb_cls = self.verb_cls.repeat(bs, 1).unsqueeze(0)
        noun_cls = self.noun_cls.repeat(bs, 1).unsqueeze(0)

        semantic_feat = semantic_feat.transpose(0, 1)  # [t, bs, hidden_dim]

        verb_attn_output, _ = self.multihead_attn(verb_cls, semantic_feat, semantic_feat)
        verb_attn_output = verb_attn_output.squeeze(0)  # [bs, hidden_dim]

        noun_attn_output, _ = self.multihead_attn(noun_cls, semantic_feat, semantic_feat)
        noun_attn_output = noun_attn_output.squeeze(0)  # [bs, hidden_dim]

        verb_logits = self.verb_classifier(verb_attn_output)
        noun_logits = self.noun_classifier(noun_attn_output)

        return verb_logits, noun_logits


# class Semantic_Decoder(nn.Module):
#     def __init__(self, input_dim, class_num):
#         super(Semantic_Decoder, self).__init__()
#         self.linear = nn.Linear(input_dim, input_dim)
#         self.event_classifier = nn.Linear(input_dim, class_num)
#
#     def forward(self, input_feat):
#         input_feat = self.linear(input_feat)
#         input_feat = input_feat.mean(1)
#         class_logits = self.event_classifier(input_feat)
#         return class_logits


class AV_VQVAE_Encoder(nn.Module):
    def __init__(self, video_dim, audio_dim, video_output_dim, audio_output_dim, n_embeddings, embedding_dim):
        super(AV_VQVAE_Encoder, self).__init__()
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.hidden_dim = embedding_dim
        self.Audio_encoder = Audio_Encoder(audio_dim, audio_output_dim)
        self.Video_encoder = Video_Encoder(video_dim, video_output_dim)
        self.audio_vib = VIB(self.hidden_dim)
        self.video_vib = VIB(self.hidden_dim)
        self.W2 = True
        # self.semantic_perceiver = Mask_MHCrossAttention(video_dim, audio_dim, embedding_dim, num_heads=6 ,head_dim=embedding_dim
        self.n_mu = 7
        self.em_iter = 5
        self.vid_mu = nn.Parameter(torch.randn(self.n_mu, self.hidden_dim))
        self.aud_mu = nn.Parameter(torch.randn(self.n_mu, self.hidden_dim))
        # self.vid_remain = nn.Parameter(torch.randn(self.n_mu, video_output_dim))
        # self.aud_remain = nn.Parameter(torch.randn(self.n_mu, audio_output_dim))
        # self.positional_embedding = nn.Parameter(torch.empty((10, 1, self.hidden_dim), requires_grad=True))

        # self.moment_Cross_quantizer = Cross_VQEmbeddingEMA(n_embeddings, self.hidden_dim)
        self.Cross_quantizer = Cross_VQEmbeddingEMA(n_embeddings, self.hidden_dim)
        self.Cross_MB = Cross_PCLEMA(n_embeddings, self.hidden_dim)

        self.video_semantic_encoder = Video_Semantic_Encoder(video_dim)
        self.video_self_att = InternalTemporalRelationModule(input_dim=video_dim, d_model=self.hidden_dim)
        self.audio_self_att = InternalTemporalRelationModule(input_dim=audio_dim, d_model=self.hidden_dim)

    # def Audio_VQ_Encoder(self, audio_feat):
    #     audio_feat = audio_feat.cuda()
    #     bs=audio_feat.shape[0]
    #     audio_aggregated_embed = self.aud_mu.repeat(bs, 1, 1)
    #     audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
    #     audio_semantic_result = self.audio_self_att(audio_semantic_result)
    #     audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()
    #     (audio_mu, audio_std), audio_semantic_bottom = self.audio_vib(audio_semantic_result)
    #     audio_aggregated = self.EM_RBF(audio_aggregated_embed, audio_semantic_bottom, self.em_iter)
    #     audio_entity = torch.cat((torch.mean(audio_semantic_bottom, dim=1, keepdim=True),
    #                              torch.mean(audio_aggregated, dim=1, keepdim=True)), dim=1)
    #     audio_entity = self.Cross_quantizer.Audio_vq_embedding(audio_entity)
    #
    #     return audio_entity

    # def Video_VQ_Encoder(self, video_feat):
    #     video_feat = video_feat.cuda()
    #     bs = video_feat.shape[0]
    #     visual_aggregated_embed = self.vid_mu.repeat(bs, 1, 1)
    #     video_semantic_result = video_feat.transpose(0, 1).contiguous()
    #     video_semantic_result = self.video_self_att(video_semantic_result)
    #     video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
    #     (video_mu, video_std), video_semantic_bottom = self.video_vib(video_semantic_result)
    #     video_aggregated = self.EM_RBF(visual_aggregated_embed, video_semantic_bottom, self.em_iter)
    #     video_entity = torch.cat((torch.mean(video_semantic_bottom, dim=1, keepdim=True),
    #                              torch.mean(video_aggregated, dim=1, keepdim=True)), dim=1)
    #     video_entity = self.Cross_quantizer.Video_vq_embedding(video_entity)
    #
    #     return video_entity

    def Audio_VQ_Encoder(self, audio_feat):
        audio_feat = audio_feat.cuda()
        bs = audio_feat.shape[0]
        audio_aggregated_embed = self.aud_mu.repeat(bs, 1, 1)
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()
        (audio_mu, audio_std), audio_semantic_bottom = self.audio_vib(audio_semantic_result)
        audio_aggregated, a_latent = self.EM_RBF(audio_aggregated_embed, audio_semantic_bottom, self.em_iter)
        # audio_moment_vq = self.moment_Cross_quantizer.Audio_vq_embedding(audio_aggregated)
        audio_vq = self.Cross_quantizer.Audio_vq_embedding(audio_aggregated)
        # audio_vq = torch.cat((audio_moment_vq, audio_entity), dim=1)

        return audio_vq

    def Video_VQ_Encoder(self, video_feat):
        video_feat = video_feat.cuda()
        bs = video_feat.shape[0]
        visual_aggregated_embed = self.vid_mu.repeat(bs, 1, 1)
        video_semantic_result = video_feat.transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        (video_mu, video_std), video_semantic_bottom = self.video_vib(video_semantic_result)
        video_aggregated, v_latent = self.EM_RBF(visual_aggregated_embed, video_semantic_bottom, self.em_iter)
        # video_moment_vq = self.moment_Cross_quantizer.Video_vq_embedding(video_aggregated)
        video_vq = self.Cross_quantizer.Video_vq_embedding(video_aggregated)
        # video_vq = torch.cat((video_moment_vq, video_entity), dim=1)

        return video_vq

    def Video_forward(self, video_feat):
        video_feat = video_feat.cuda()
        bs = video_feat.shape[0]
        visual_aggregated_embed = self.vid_mu.repeat(bs, 1, 1)
        video_semantic_result = video_feat.transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        (video_mu, video_std), video_semantic_bottom = self.video_vib(video_semantic_result)
        video_aggregated, v_latent = self.EM_RBF(visual_aggregated_embed, video_semantic_bottom, self.em_iter)

        return video_aggregated

    def Audio_forward(self, audio_feat):
        audio_feat = audio_feat.cuda()
        bs = audio_feat.shape[0]
        audio_aggregated_embed = self.aud_mu.repeat(bs, 1, 1)
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()
        (audio_mu, audio_std), audio_semantic_bottom = self.audio_vib(audio_semantic_result)
        audio_aggregated, a_latent = self.EM_RBF(audio_aggregated_embed, audio_semantic_bottom, self.em_iter)

        return audio_aggregated

    def estimate_gaussian_params(self, encodings):
        """
        从高斯分布的均值和采样中计算其标准差
        mu: (batch_size, 1, encoding_dim) Tensor
        encodings: (batch_size, sequence_length, encoding_dim) Tensor
        返回:
            sigma: (batch_size, sequence_length, encoding_dim, encoding_dim) Tensor
        """
        batch_size, seq_len, encoding_dim = encodings.size()

        # 减去均值
        mu = torch.mean(encodings, dim=1, keepdim=True)
        centered = encodings - mu

        # covariance
        sigma = (1.0 / (encodings.shape[1])) * torch.bmm(centered.permute(0, 2, 1), centered)

        return sigma

    def compute_laplacian(self, x, n_neighbors=5):
        '''
        计算图拉普拉斯矩阵
        x: 输入数据 [b, l, d]
        n_neighbors: 邻居数量
        '''
        b, l, d = x.size()
        # device = x.device()
        x = x.view(b * l, d).detach().cpu().numpy()
        adj_matrix = kneighbors_graph(x, n_neighbors, mode='connectivity', include_self=True)
        adj_matrix = adj_matrix.toarray()  # 转换为稠密矩阵
        degree_matrix = np.diag(adj_matrix.sum(axis=1))  # 计算度矩阵
        laplacian_matrix = degree_matrix - adj_matrix  # 计算图拉普拉斯矩阵
        return torch.tensor(laplacian_matrix, dtype=torch.float64).cuda()

    # def EM_RBF(self, mu, x, iter, lambda_m = 0.01, n_neighbors=5):
    #     '''
    #     mu [b,k,d]
    #     x  [b,l,d]
    #     lambda_m: 流形正则化系数
    #     n_neighbors: 邻居数量
    #     '''
    #     em_iter = iter
    #     # propagation -> make mu as video-specific mu
    #     norm_x = self.calculate_l1_norm(x)
    #
    #     latent = mu
    #     for _ in range(em_iter):
    #         norm_mu = self.calculate_l1_norm(mu)
    #         sigma = 1.2
    #         latent_z = F.softmax(-0.5 * ((norm_mu[:, :, None, :] - norm_x[:, None, :, :]) ** 2).sum(-1) / sigma ** 2,
    #                              dim=1)
    #         # print("latent_z",latent_z.shape)
    #         norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True) + 1e-9)
    #         updated_mu = torch.einsum('nkt,ntd->nkd', [norm_latent_z, x])
    #         latent = torch.cat((latent, updated_mu), dim=1)
    #         # manifold regularization
    #     b, k, d = updated_mu.size()
    #     updated_mu_flat = updated_mu.view(b * k, d)
    #     laplacian_matrix = self.compute_laplacian(updated_mu, n_neighbors)
    #     manifold_loss = lambda_m * torch.trace(updated_mu_flat.T @ laplacian_matrix @ updated_mu_flat)
    #     return manifold_loss, mu, latent

    def EM_RBF(self, mu, x, iter):
        '''
        mu [b,k,d]
        x  [b,l,d]
        '''
        em_iter = iter
        # propagation -> make mu as video-specific mu
        norm_x = self.calculate_l1_norm(x)
        latent = None
        for _ in range(em_iter):
            norm_mu = self.calculate_l1_norm(mu)
            sigma = 1.2
            latent_z = F.softmax(-0.5 * ((norm_mu[:, :, None, :] - norm_x[:, None, :, :]) ** 2).sum(-1) / sigma ** 2,
                                 dim=1)
            # print("latent_z",latent_z.shape)  torch.Size([80, 7, 15])
            norm_latent_z = latent_z / (latent_z.sum(dim=-1, keepdim=True) + 1e-9)
            mu = torch.einsum('nkt,ntd->nkd', [norm_latent_z, x])
            if latent != None:
                latent = torch.cat((latent, torch.mean(mu,dim=1, keepdim=True)), dim=1)
            else:
                latent = torch.mean(mu,dim=1, keepdim=True)
        return mu, latent

    def calculate_l1_norm(self, f):
        f_norm = torch.norm(f, p=1, dim=-1, keepdim=True)
        f = f / (f_norm + 1e-9)
        return f

    def wasserstein_distance_unit_variance(self, mu1, mu2):
        # 计算均值向量的欧几里得距离
        return np.linalg.norm(mu1 - mu2)

    def kl_divergence_unit_variance(self, mu1, mu2):
        # 计算均值向量的欧几里得距离的平方
        mean_diff_squared = torch.sum((mu1 - mu2) ** 2)

        # 计算KL散度
        kl_div = 0.5 * mean_diff_squared

        return kl_div

    def compute_fine_matrix_slice(self, featA, featB):
        score_matrix = torch.einsum('atd,bvd->abtv', [featA, featB])
        return score_matrix

    def contrastive_loss(self, score_matrix):  ### labels for unicl
        temp = 1.0
        score_matrix = torch.mean(score_matrix, dim=(-1, -2), keepdim=False)
        score_matrix = score_matrix / temp
        matrix1 = -F.log_softmax(score_matrix, dim=1)
        matrix2 = -F.log_softmax(score_matrix, dim=0)
        loss1 = matrix1.diag()
        loss2 = matrix2.diag()
        contra_loss = torch.mean(torch.cat((loss1, loss2), dim=0))

        return contra_loss

    def forward(self, audio_feat, video_feat, epoch):
        video_feat = video_feat.cuda()
        audio_feat = audio_feat.cuda()
        bs = video_feat.shape[0]
        video_aggregated_embed = self.vid_mu.repeat(bs, 1, 1)
        audio_aggregated_embed = self.aud_mu.repeat(bs, 1, 1)
        # position_embedding = self.positional_embedding.repeat(1, bs, 1)
        video_encoder_result = self.Video_encoder(video_feat)  # [batch, length, video_output_dim]
        audio_encoder_result = self.Audio_encoder(audio_feat)  # [batch, length, audio_output_dim]

        kl_divergence = self.kl_divergence_unit_variance(self.vid_mu, self.aud_mu)

        video_semantic_result = video_feat.transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]

        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]

        (video_mu, video_std), video_semantic_bottom = self.video_vib(video_semantic_result)
        (audio_mu, audio_std), audio_semantic_bottom = self.audio_vib(audio_semantic_result)
        video_aggregated_embed, v_latent = self.EM_RBF(video_aggregated_embed, video_semantic_bottom, self.em_iter)
        audio_aggregated_embed, a_latent = self.EM_RBF(audio_aggregated_embed, audio_semantic_bottom, self.em_iter)

        kl_divergence /= bs

        # audio_vq, video_vq, a_loss, v_loss, _, _, cmcm_loss, code_num \
        #     = self.Cross_quantizer(audio_aggregated_embed, video_aggregated_embed, epoch)
        if epoch<10:
            score_matrix = self.compute_fine_matrix_slice(video_aggregated_embed, audio_aggregated_embed)
            cpcl_loss = self.contrastive_loss(score_matrix)
        else:
            cpcl_loss = self.Cross_MB(audio_aggregated_embed, video_aggregated_embed, epoch)

        return video_mu, video_std, video_aggregated_embed, video_semantic_bottom, \
               audio_mu, audio_std, audio_aggregated_embed, audio_semantic_bottom, \
               audio_semantic_result, video_semantic_result, audio_encoder_result, video_encoder_result, \
               cpcl_loss, kl_divergence


class AV_VQVAE_Decoder(nn.Module):
    def __init__(self, video_dim, audio_dim, video_output_dim, audio_output_dim, embedding_dim):
        super(AV_VQVAE_Decoder, self).__init__()
        self.hidden_dim = embedding_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.video_output_dim = video_output_dim
        self.audio_output_dim = audio_output_dim
        self.Audio_decoder = Audio_Decoder(audio_output_dim, audio_dim, self.hidden_dim)
        self.Video_decoder = Video_Decoder(video_output_dim, video_dim, self.hidden_dim)
        self.video_semantic_decoder = Semantic_Decoder(self.hidden_dim, class_num=106)
        self.audio_semantic_decoder = Semantic_Decoder(self.hidden_dim, class_num=141)

    def forward(self, audio_feat, video_feat, audio_encoder_result, video_encoder_result, audio_vq, video_vq):
        video_feat = video_feat.cuda()
        audio_feat = audio_feat.cuda()
        video_recon_result = self.Video_decoder(video_encoder_result, video_vq)
        audio_recon_result = self.Audio_decoder(audio_encoder_result, audio_vq)
        video_recon_loss = F.mse_loss(video_recon_result, video_feat)
        audio_recon_loss = F.mse_loss(audio_recon_result, audio_feat)
        video_class = self.video_semantic_decoder(video_vq)
        audio_class = self.audio_semantic_decoder(audio_vq)

        return video_recon_loss, audio_recon_loss, video_class, audio_class


class Cross_VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(Cross_VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.growth_threshold = 200
        self.growth_factor = 1
        self.coefficients = nn.Parameter(torch.randn(1), requires_grad=True)

        init_bound = 1 / 400
        embedding = torch.Tensor(n_embeddings, embedding_dim)

        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())
        self.register_buffer("unactivated_count", -torch.ones(n_embeddings))

    def expand_embeddings(self, num_new_embeddings, input_data):
        embedding_dim = self.embedding.size(1)
        device = self.embedding.device
        new_embeddings = input_data[:num_new_embeddings] + torch.Tensor(num_new_embeddings, embedding_dim).uniform_(
            -1 / 1024, 1 / 1024).to(device)

        new_ema_count = torch.zeros(num_new_embeddings).to(device)
        new_ema_weight = new_embeddings.clone().to(device)
        new_unactivated_count = -torch.ones(num_new_embeddings).to(device)

        self.embedding = torch.cat([self.embedding, new_embeddings], dim=0)
        self.ema_count = torch.cat([self.ema_count, new_ema_count], dim=0)
        self.ema_weight = torch.cat([self.ema_weight, new_ema_weight], dim=0)
        self.unactivated_count = torch.cat([self.unactivated_count, new_unactivated_count], dim=0)

    def check_and_expand_embeddings(self, distances, threshold, input_data):
        min_distances, min_indices = torch.min(distances, dim=-1)
        # print(min_distances)
        indices_to_expand = min_distances > threshold
        num_new_embeddings = indices_to_expand.sum().item()
        if num_new_embeddings > 0:
            num_new_embeddings = int(num_new_embeddings * self.growth_factor)
            new_data = input_data[indices_to_expand][:num_new_embeddings]
            self.expand_embeddings(num_new_embeddings, new_data)

    def Audio_vq_embedding(self, audio_semantic):

        B, T, D = audio_semantic.size()
        # audio_semantic = F.normalize(audio_semantic, p=1, dim=-1)
        a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        a_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                  a_flat, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]
        print(torch.mean(a_distances))
        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT,1]
        a_quantized = F.embedding(a_indices, self.embedding)
        a_quantized = a_quantized.view_as(audio_semantic)  # [BxT,D]->[B,T,D]
        a_quantized = audio_semantic + (a_quantized - audio_semantic).detach()
        return a_quantized

    def Video_vq_embedding(self, video_semantic):

        B, T, D = video_semantic.size()
        # video_semantic = F.normalize(video_semantic, p=1, dim=-1)
        v_flat = video_semantic.detach().reshape(-1, D)
        v_distance = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                 torch.sum(v_flat ** 2, dim=1, keepdim=True),
                                 v_flat, self.embedding.t(),
                                 alpha=-2.0, beta=1.0)
        print(torch.mean(v_distance))
        v_indices = torch.argmin(v_distance.double(), dim=-1)
        v_quantized = F.embedding(v_indices, self.embedding)
        v_quantized = v_quantized.view_as(video_semantic)
        v_quantized = video_semantic + (v_quantized - video_semantic).detach()
        return v_quantized

    def calculate_entropy(self, prob_dist):
        return -torch.sum(prob_dist * torch.log(prob_dist + 1e-5), dim=-1)

    def forward(self, audio_semantic, video_semantic, epoch):
        B, T, D = audio_semantic.size()
        # x_flat = x.detach().reshape(-1, D) #x:[B,T,D]->x_flat:[BxT,D]
        a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        v_flat = video_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]

        # a_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
        #                           torch.sum(a_flat ** 2, dim=1, keepdim=True),
        #                           a_flat, self.embedding.t(),
        #                           alpha=-2.0, beta=1.0)  # [BxT, M]
        #
        # v_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
        #                           torch.sum(v_flat ** 2, dim=1, keepdim=True),
        #                           v_flat, self.embedding.t(),
        #                           alpha=-2.0, beta=1.0)  # [BxT, M]
        # if epoch>10:
        # self.check_and_expand_embeddings(a_distances, self.growth_threshold, a_flat+v_flat)
        # self.check_and_expand_embeddings(v_distances, self.growth_threshold, v_flat)

        M, D = self.embedding.size()
        a_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                  a_flat, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]

        v_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(v_flat ** 2, dim=1, keepdim=True),
                                  v_flat, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]

        a_distances_gradient = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                           torch.sum(audio_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                           audio_semantic.reshape(-1, D), self.embedding.t(),
                                           alpha=-2.0, beta=1.0)  # [BxT, M]

        v_distances_gradient = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                           torch.sum(video_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                           video_semantic.reshape(-1, D), self.embedding.t(),
                                           alpha=-2.0, beta=1.0)  # [BxT, M]

        a_ph = F.softmax(-torch.sqrt(a_distances_gradient), dim=1)  # [BxT, M]
        v_ph = F.softmax(-torch.sqrt(v_distances_gradient), dim=1)  # [BxT, M]
        # a_adjustment = a_ph
        # v_adjustment = v_ph
        a_entropy = self.calculate_entropy(a_ph)  # [BxT]
        v_entropy = self.calculate_entropy(v_ph)  # [BxT]
        a_ph = torch.reshape(a_ph, ((B, T, M)))  # [BxT, M] -> [B, T, M]
        a_consistency_loss = torch.sum(torch.abs(a_ph - torch.mean(a_ph, dim=1, keepdim=True)))
        a_pH = torch.mean(a_ph, dim=1)  # [B, T, M] -> [B, M]
        v_ph = torch.reshape(v_ph, ((B, T, M)))  # [BxT, M] -> [B, T, M]
        v_consistency_loss = torch.sum(torch.abs(v_ph - torch.mean(v_ph, dim=1, keepdim=True)))
        v_pH = torch.mean(v_ph, dim=1)  # [B, T, M] -> [B, M]

        Scode = a_pH @ torch.log(v_pH.t() + 1e-10) + v_pH @ torch.log(a_pH.t() + 1e-10)

        MaxScode = torch.max(-Scode)
        EScode = torch.exp(Scode + MaxScode)

        EScode_sumdim1 = torch.sum(EScode, dim=1)

        Lcmcm = 0
        for i in range(B):
            Lcmcm -= torch.log(EScode[i, i] / (EScode_sumdim1[i] + self.epsilon))
        Lcmcm /= B
        if epoch<10:
            Lcmcm*=0
        a_consistency_loss /= B
        v_consistency_loss /= B

        # print(Lcmcm, a_consistency_loss, v_consistency_loss)

        # Entropy normalization
        max_entropy = np.log(self.embedding.size(0))
        a_adjustment = 1 - a_entropy / max_entropy  # [BxT]
        v_adjustment = 1 - v_entropy / max_entropy  # [BxT]

        a_adjustment = a_adjustment.unsqueeze(-1)  # [BxT, 1]
        v_adjustment = v_adjustment.unsqueeze(-1)  # [BxT, 1]

        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT,1]
        a_encodings = F.one_hot(a_indices, M).double()  # [BxT, M]
        a_quantized = F.embedding(a_indices, self.embedding)
        a_quantized = a_quantized.view_as(audio_semantic)  # [BxT,D]->[B,T,D]

        v_indices = torch.argmin(v_distances.double(), dim=-1)  # [BxT,1]
        v_encodings = F.one_hot(v_indices, M).double()  # [BxT, M]
        v_quantized = F.embedding(v_indices, self.embedding)
        v_quantized = v_quantized.view_as(video_semantic)  # [BxT,D]->[B,T,D]

        if True:
            a_indices_reshape = a_indices.reshape(B, T)
            v_indices_reshape = v_indices.reshape(B, T)
            a_indices_mode = torch.mode(a_indices_reshape, dim=-1, keepdim=False)
            v_indices_mode = torch.mode(v_indices_reshape, dim=-1, keepdim=False)

            equal_item = (a_indices_mode.values == v_indices_mode.values)
            equal_num = equal_item.sum()

        if True:
            a_encodings = a_adjustment * a_encodings  # Adjust the encodings
            v_encodings = v_adjustment * v_encodings  # Adjust the encodings

            a_encodings = a_encodings.clone().detach()  # Adjust the encodings
            v_encodings = v_encodings.clone().detach()  # Adjust the encodings

        # Softmax for coefficients
        weights = torch.sigmoid(self.coefficients)  # [1]

        # video
        self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(v_encodings, dim=0)
        n = torch.sum(self.ema_count)
        self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
        v_dw = torch.matmul(v_encodings.t(), v_flat)
        # ********************************************************
        va_dw = torch.matmul(v_encodings.t(), a_flat)
        # ********************************************************
        # self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * v_dw
        self.ema_weight = self.decay * self.ema_weight + \
                          0.5 * (1 - self.decay) * v_dw + 0.5 * (1 - self.decay) * va_dw
        self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        # audio
        self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(a_encodings, dim=0)
        n = torch.sum(self.ema_count)
        self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
        # print(a_encodings.shape:[400, 600], a_flat.shape:[400, 256], a_adjustment.shape:[400, 1])
        a_dw = torch.matmul(a_encodings.t(), a_flat)
        # ********************************************************
        av_dw = torch.matmul(a_encodings.t(), v_flat)
        # ********************************************************
        # self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * a_dw
        self.ema_weight = self.decay * self.ema_weight + \
                          0.5 * (1 - self.decay) * a_dw + 0.5 * (1 - self.decay) * av_dw
        self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)



        self.unactivated_count += 1
        for indice in a_indices:
            self.unactivated_count[indice.item()] = 0
        for indice in v_indices:
            self.unactivated_count[indice.item()] = 0
        activated_indices = []
        unactivated_indices = []
        for i, x in enumerate(self.unactivated_count):
            if x > 300:
                unactivated_indices.append(i)
                self.unactivated_count[i] = 0
            elif x >= 0 and x < 100:
                activated_indices.append(i)
        activated_quantized = F.embedding(torch.tensor(activated_indices, dtype=torch.int32).cuda(), self.embedding)
        for i in unactivated_indices:
            self.embedding[i] = activated_quantized[random.randint(0, len(activated_indices) - 1)] + torch.Tensor(
                D).uniform_(-1 / 1024, -1 / 1024).cuda()

        cmcm_loss = 0.5 * Lcmcm

        a_e_latent_loss = F.mse_loss(audio_semantic, a_quantized.detach())
        av_e_latent_loss = F.mse_loss(audio_semantic, v_quantized.detach())
        # a_loss = self.commitment_cost * 1.0 * a_e_latent_loss
        a_loss = self.commitment_cost * 2.0 * a_e_latent_loss + self.commitment_cost * av_e_latent_loss
        v_e_latent_loss = F.mse_loss(video_semantic, v_quantized.detach())
        va_e_latent_loss = F.mse_loss(video_semantic, a_quantized.detach())
        # v_loss = self.commitment_cost * 1.0 * v_e_latent_loss
        v_loss = self.commitment_cost * 2.0 * v_e_latent_loss + self.commitment_cost * va_e_latent_loss

        a_quantized = audio_semantic + (a_quantized - audio_semantic).detach()
        v_quantized = video_semantic + (v_quantized - video_semantic).detach()

        a_avg_probs = torch.mean(a_encodings, dim=0)
        a_perplexity = torch.exp(-torch.sum(a_avg_probs * torch.log(a_avg_probs + 1e-10)))
        v_avg_probs = torch.mean(v_encodings, dim=0)
        v_perplexity = torch.exp(-torch.sum(v_avg_probs * torch.log(v_avg_probs + 1e-10)))
        return a_quantized, v_quantized, a_consistency_loss, v_consistency_loss, a_perplexity, v_perplexity, cmcm_loss, M


class Cross_PCLEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(Cross_PCLEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 400
        embedding = torch.Tensor(n_embeddings, embedding_dim)

        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())
        self.register_buffer("unactivated_count", -torch.ones(n_embeddings))

    def Audio_vq_embedding(self, audio_semantic):

        B, T, D = audio_semantic.size()
        # audio_semantic = F.normalize(audio_semantic, p=1, dim=-1)
        a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        a_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                  a_flat, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]
        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT,1]
        a_quantized = F.embedding(a_indices, self.embedding)
        a_quantized = a_quantized.view_as(audio_semantic)  # [BxT,D]->[B,T,D]
        a_quantized = audio_semantic + (a_quantized - audio_semantic).detach()
        return a_quantized

    def Video_vq_embedding(self, video_semantic):

        B, T, D = video_semantic.size()
        # video_semantic = F.normalize(video_semantic, p=1, dim=-1)
        v_flat = video_semantic.detach().reshape(-1, D)
        v_distance = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                 torch.sum(v_flat ** 2, dim=1, keepdim=True),
                                 v_flat, self.embedding.t(),
                                 alpha=-2.0, beta=1.0)
        v_indices = torch.argmin(v_distance.double(), dim=-1)
        v_quantized = F.embedding(v_indices, self.embedding)
        v_quantized = v_quantized.view_as(video_semantic)
        v_quantized = video_semantic + (v_quantized - video_semantic).detach()
        return v_quantized

    def calculate_entropy(self, prob_dist):
        return -torch.sum(prob_dist * torch.log(prob_dist + 1e-5), dim=-1)

    def info_nce_loss(self, features, self_positive_indices, coupled_positive_indices, temperature=0.1):

        # 计算特征和所有嵌入向量之间的相似度
        all_similarity = F.cosine_similarity(features.unsqueeze(2), self.embedding.unsqueeze(1), dim=2)  # (560, 256, 1),(600, 1, 256)

        # 应用温度缩放
        all_similarity /= temperature

        # 计算损失
        self_pcl_loss = F.cross_entropy(all_similarity, self_positive_indices)
        coupled_pcl_loss = F.cross_entropy(all_similarity, coupled_positive_indices)
        cpcl_loss = self.commitment_cost*self_pcl_loss+(1-self.commitment_cost)*coupled_pcl_loss

        return cpcl_loss

    def forward(self, audio_semantic, video_semantic, epoch):
        B, T, D = audio_semantic.size()
        a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        v_flat = video_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]

        M, D = self.embedding.size()
        a_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                  a_flat, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]

        v_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(v_flat ** 2, dim=1, keepdim=True),
                                  v_flat, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]

        a_distances_gradient = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                           torch.sum(audio_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                           audio_semantic.reshape(-1, D), self.embedding.t(),
                                           alpha=-2.0, beta=1.0)  # [BxT, M]

        v_distances_gradient = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                           torch.sum(video_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                           video_semantic.reshape(-1, D), self.embedding.t(),
                                           alpha=-2.0, beta=1.0)  # [BxT, M]

        a_ph = F.softmax(-torch.sqrt(a_distances_gradient), dim=1)  # [BxT, M]
        v_ph = F.softmax(-torch.sqrt(v_distances_gradient), dim=1)  # [BxT, M]
        a_entropy = self.calculate_entropy(a_ph)  # [BxT]
        v_entropy = self.calculate_entropy(v_ph)  # [BxT]
        a_ph = torch.reshape(a_ph, ((B, T, M)))  # [BxT, M] -> [B, T, M]
        a_consistency_loss = torch.sum(torch.abs(a_ph - torch.mean(a_ph, dim=1, keepdim=True)))
        a_pH = torch.mean(a_ph, dim=1)  # [B, T, M] -> [B, M]
        v_ph = torch.reshape(v_ph, ((B, T, M)))  # [BxT, M] -> [B, T, M]
        v_consistency_loss = torch.sum(torch.abs(v_ph - torch.mean(v_ph, dim=1, keepdim=True)))
        v_pH = torch.mean(v_ph, dim=1)  # [B, T, M] -> [B, M]

        Scode = a_pH @ torch.log(v_pH.t() + 1e-10) + v_pH @ torch.log(a_pH.t() + 1e-10)

        MaxScode = torch.max(-Scode)
        EScode = torch.exp(Scode + MaxScode)

        EScode_sumdim1 = torch.sum(EScode, dim=1)

        Lcmcm = 0
        for i in range(B):
            Lcmcm -= torch.log(EScode[i, i] / (EScode_sumdim1[i] + self.epsilon))
        Lcmcm /= B
        a_consistency_loss /= B
        v_consistency_loss /= B

        # Entropy normalization
        max_entropy = np.log(self.embedding.size(0))
        a_adjustment = 1 - a_entropy / max_entropy  # [BxT]
        v_adjustment = 1 - v_entropy / max_entropy  # [BxT]

        a_adjustment = a_adjustment.unsqueeze(-1)  # [BxT, 1]
        v_adjustment = v_adjustment.unsqueeze(-1)  # [BxT, 1]

        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT,1]
        a_encodings = F.one_hot(a_indices, M).double()  # [BxT, M]
        a_quantized = F.embedding(a_indices, self.embedding)
        a_quantized = a_quantized.view_as(audio_semantic)  # [BxT,D]->[B,T,D]

        v_indices = torch.argmin(v_distances.double(), dim=-1)  # [BxT,1]
        v_encodings = F.one_hot(v_indices, M).double()  # [BxT, M]
        v_quantized = F.embedding(v_indices, self.embedding)
        v_quantized = v_quantized.view_as(video_semantic)  # [BxT,D]->[B,T,D]

        a_positive = a_encodings
        v_positive = v_encodings

        if True:
            a_indices_reshape = a_indices.reshape(B, T)
            v_indices_reshape = v_indices.reshape(B, T)
            a_indices_mode = torch.mode(a_indices_reshape, dim=-1, keepdim=False)
            v_indices_mode = torch.mode(v_indices_reshape, dim=-1, keepdim=False)

            equal_item = (a_indices_mode.values == v_indices_mode.values)
            equal_num = equal_item.sum()

        if True:
            a_encodings = a_adjustment * a_encodings  # Adjust the encodings
            v_encodings = v_adjustment * v_encodings  # Adjust the encodings

            a_encodings = a_encodings.clone().detach()  # Adjust the encodings
            v_encodings = v_encodings.clone().detach()  # Adjust the encodings

        # video
        self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(v_encodings, dim=0)
        n = torch.sum(self.ema_count)
        self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
        v_dw = torch.matmul(v_encodings.t(), v_flat)
        va_dw = torch.matmul(v_encodings.t(), a_flat)
        # ********************************************************
        self.ema_weight = self.decay * self.ema_weight + \
                          0.5 * (1 - self.decay) * v_dw + 0.5 * (1 - self.decay) * va_dw
        self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        # audio
        self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(a_encodings, dim=0)
        n = torch.sum(self.ema_count)
        self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
        # print(a_encodings.shape:[400, 600], a_flat.shape:[400, 256], a_adjustment.shape:[400, 1])
        a_dw = torch.matmul(a_encodings.t(), a_flat)
        av_dw = torch.matmul(a_encodings.t(), v_flat)
        # ********************************************************
        self.ema_weight = self.decay * self.ema_weight + \
                          0.5 * (1 - self.decay) * a_dw + 0.5 * (1 - self.decay) * av_dw
        self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)



        self.unactivated_count += 1
        for indice in a_indices:
            self.unactivated_count[indice.item()] = 0
        for indice in v_indices:
            self.unactivated_count[indice.item()] = 0
        activated_indices = []
        unactivated_indices = []
        for i, x in enumerate(self.unactivated_count):
            if x > 300:
                unactivated_indices.append(i)
                self.unactivated_count[i] = 0
            elif x >= 0 and x < 100:
                activated_indices.append(i)
        activated_quantized = F.embedding(torch.tensor(activated_indices, dtype=torch.int32).cuda(), self.embedding)
        for i in unactivated_indices:
            self.embedding[i] = activated_quantized[random.randint(0, len(activated_indices) - 1)] + torch.Tensor(
                D).uniform_(-1 / 1024, -1 / 1024).cuda()

        cmcm_loss = 0.5 * Lcmcm

        a_cpcl_loss = self.info_nce_loss(audio_semantic.reshape(-1, D), a_positive, v_positive)
        v_cpcl_loss = self.info_nce_loss(video_semantic.reshape(-1, D), v_positive, a_positive)
        cpcl_loss = self.commitment_cost * a_cpcl_loss + self.commitment_cost * v_cpcl_loss
        cpcl_loss/=B

        return cpcl_loss