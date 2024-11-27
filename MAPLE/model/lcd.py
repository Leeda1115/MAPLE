import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionModel(nn.Module):
    def __init__(self, latent_dim):
        super(DiffusionModel, self).__init__()
        self.latent_dim = latent_dim

        # Cross Attention layer weights
        self.fc_query = nn.Linear(latent_dim, latent_dim)
        self.fc_key = nn.Linear(latent_dim, latent_dim)
        self.fc_value = nn.Linear(latent_dim, latent_dim)

    def cross_attention(self, x, condition):
        """
        Perform cross-attention between x and condition.
        """
        query = self.fc_query(x)
        key = self.fc_key(condition)
        value = self.fc_value(condition)

        # Compute attention weights
        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention weights to value
        attended_values = torch.matmul(attn_weights, value)

        return attended_values

    def forward(self, input_seq, condition_seq):
        # Cross Attention
        attended_values = self.cross_attention(input_seq, condition_seq)

        # Fusion with input sequence representation
        fused_rep = torch.cat([latent_rep, attended_values], dim=-1)

        return fused_rep


class LatentDiffusionModel(nn.Module):
    def __init__(self, diffusion_model, latent_dim):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.latent_dim = latent_dim

    def forward(self, x_info, c_info):
        x = x_info['x']
        t = torch.randint(0, self.diffusion_model.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x_info, t, c_info)

    def p_losses(self, x_info, t, c_info, noise=None):
        x = x_info['x']
        noise = torch.randn_like(x) if noise is None else noise
        x_noisy = self.diffusion_model.q_sample(x_start=x, t=t, noise=noise)
        x_info['x'] = x_noisy
        model_output = self.apply_model(x_info, t, c_info)

        loss_dict = {}

        if self.parameterization == "x0":
            target = x
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        bs = model_output.shape[0]
        loss_simple = self.diffusion_model.get_loss(model_output, target, mean=False).view(bs, -1).mean(-1)
        loss_dict['loss_simple'] = loss_simple.mean()

        logvar_t = self.diffusion_model.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        if self.diffusion_model.learn_logvar:
            loss_dict['loss_gamma'] = loss.mean()
            loss_dict['logvar'] = self.diffusion_model.logvar.data.mean()

        loss = self.diffusion_model.l_simple_weight * loss.mean()

        loss_vlb = self.diffusion_model.get_loss(model_output, target, mean=False).view(bs, -1).mean(-1)
        loss_vlb = (self.diffusion_model.lvlb_weights[t] * loss_vlb).mean()
        loss_dict['loss_vlb'] = loss_vlb
        loss_dict.update({'Loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def apply_model(self, x_info, timesteps, c_info):
        x_type, x = x_info['type'], x_info['x']
        c_type, c = c_info['type'], c_info['c']
        dtype = x.dtype

        hs = []

        from .openaimodel import timestep_embedding

        glayer_ptr = x_type if self.diffusion_model.global_layer_ptr is None else self.diffusion_model.global_layer_ptr
        model_channels = self.diffusion_model.diffuser[glayer_ptr].model_channels
        t_emb = timestep_embedding(timesteps, model_channels, repeat_only=False).to(dtype)
        emb = self.diffusion_model.diffuser[glayer_ptr].time_embed(t_emb)

        d_iter = iter(self.diffusion_model.diffuser[x_type].data_blocks)
        c_iter = iter(self.diffusion_model.diffuser[c_type].context_blocks)

        i_order = self.diffusion_model.diffuser[x_type].i_order
        m_order = self.diffusion_model.diffuser[x_type].m_order
        o_order = self.diffusion_model.diffuser[x_type].o_order

        h = x
        for ltype in i_order:
            if ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == 'c':
                module = next(c_iter)
                h = module(h, emb, c)
            elif ltype == 'save_hidden_feature':
                hs.append(h)

        for ltype in m_order:
            if ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == 'c':
                module = next(c_iter)
                h = module(h, emb, c)

        for ltype in o_order:
            if ltype == 'load_hidden_feature':
                h = torch.cat([h, hs.pop()], dim=1)
            elif ltype == 'd':
                module = next(d_iter)
                h = module(h, emb, None)
            elif ltype == 'c':
                module = next(c_iter)
                h = module(h, emb, c)
        o = h

        return o
