import torch
from torch import nn, einsum
from utils.utils import FeedForward,LayerNorm, GLANCE,FOCUS
import option

args=option.parse_args()

def exists(val):
    return val is not None


def attention(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out

def MSNSD(features,scores,bs,batch_size,drop_out,ncrops,k):
    #magnitude selection and score prediction
    features = features  # (B*10crop,32,1024)
    bc, t, f = features.size()

    scores = scores.view(bs, ncrops, -1).mean(1)  # (B,32)
    scores = scores.unsqueeze(dim=2)  # (B,32,1)

    normal_features = features[0:batch_size * 10]  # [b/2*ten,32,1024]
    normal_scores = scores[0:batch_size]  # [b/2, 32,1]

    abnormal_features = features[batch_size * 10:]
    abnormal_scores = scores[batch_size:]

    feat_magnitudes = torch.norm(features, p=2, dim=2)  # [b*ten,32]
    feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)  # [b,32]
    nfea_magnitudes = feat_magnitudes[0:batch_size]  # [b/2,32]  # normal feature magnitudes
    afea_magnitudes = feat_magnitudes[batch_size:]  # abnormal feature magnitudes
    n_size = nfea_magnitudes.shape[0]  # b/2

    if nfea_magnitudes.shape[0] == 1:  # this is for inference
        afea_magnitudes = nfea_magnitudes
        abnormal_scores = normal_scores
        abnormal_features = normal_features

    select_idx = torch.ones_like(nfea_magnitudes).cuda()
    select_idx = drop_out(select_idx)


    afea_magnitudes_drop = afea_magnitudes * select_idx
    idx_abn = torch.topk(afea_magnitudes_drop, k, dim=1)[1]
    idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])

    abnormal_features = abnormal_features.view(n_size, ncrops, t, f)
    abnormal_features = abnormal_features.permute(1, 0, 2, 3)

    total_select_abn_feature = torch.zeros(0)
    for abnormal_feature in abnormal_features:
        feat_select_abn = torch.gather(abnormal_feature, 1,
                                       idx_abn_feat)
        total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))  #

    idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])  #
    score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score),
                                dim=1)


    select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
    select_idx_normal = drop_out(select_idx_normal)
    nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
    idx_normal = torch.topk(nfea_magnitudes_drop, k, dim=1)[1]
    idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

    normal_features = normal_features.view(n_size, ncrops, t, f)
    normal_features = normal_features.permute(1, 0, 2, 3)

    total_select_nor_feature = torch.zeros(0)
    for nor_fea in normal_features:
        feat_select_normal = torch.gather(nor_fea, 1,
                                          idx_normal_feat)
        total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

    idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
    score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)

    abn_feamagnitude = total_select_abn_feature
    nor_feamagnitude = total_select_nor_feature

    return score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores

class Backbone(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        mgfn_type = 'gb',
        kernel = 5,
        dim_headnumber = 64,
        ff_repe = 4,
        dropout = 0.,
        attention_dropout = 0.
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            if mgfn_type == 'fb':
                attention = FOCUS(dim, heads = heads, dim_head = dim_headnumber, local_aggr_kernel = kernel)
            elif mgfn_type == 'gb':
                attention = GLANCE(dim, heads = heads, dim_head = dim_headnumber, dropout = attention_dropout)
            else:
                raise ValueError('unknown mhsa_type')

            self.layers.append(nn.ModuleList([
                nn.Conv1d(dim, dim, 3, padding = 1),
                attention,
                FeedForward(dim, repe = ff_repe, dropout = dropout),
            ]))

    def forward(self, x):
        for scc, attention, ff in self.layers:
            x = scc(x) + x
            x = attention(x) + x
            x = ff(x) + x

        return x

# main class

class mgfn(nn.Module):
    def __init__(
        self,
        *,
        classes=0,
        dims = (64, 128, 1024),
        depths = (args.depths1, args.depths2, args.depths3),
        mgfn_types = (args.mgfn_type1,args.mgfn_type2, args.mgfn_type3),
        lokernel = 5,
        channels = 2048,
        ff_repe = 4,
        dim_head = 64,
        dropout = 0.,
        attention_dropout = 0.
    ):
        super().__init__()
        init_dim, *_, last_dim = dims
        self.to_tokens = nn.Conv1d(channels, init_dim, kernel_size=3, stride = 1, padding = 1)

        mgfn_types = tuple(map(lambda t: t.lower(), mgfn_types))

        self.stages = nn.ModuleList([])

        for ind, (depth, mgfn_types) in enumerate(zip(depths, mgfn_types)):
            is_last = ind == len(depths) - 1
            stage_dim = dims[ind]
            heads = stage_dim // dim_head

            self.stages.append(nn.ModuleList([
                Backbone(
                    dim = stage_dim,
                    depth = depth,
                    heads = heads,
                    mgfn_type = mgfn_types,
                    ff_repe = ff_repe,
                    dropout = dropout,
                    attention_dropout = attention_dropout
                ),
                nn.Sequential(
                    LayerNorm(stage_dim),
                    nn.Conv1d(stage_dim, dims[ind + 1], 1, stride = 1),
                ) if not is_last else None
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(last_dim)
        )
        self.batch_size =  args.batch_size
        self.fc = nn.Linear(last_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(args.dropout_rate)

        self.to_mag = nn.Conv1d(1, init_dim, kernel_size=3, stride=1, padding=1)
    def forward(self, video):
        k = 3
        bs, ncrops, t, c = video.size()
        x = video.view(bs * ncrops, t, c).permute(0, 2, 1)
        x_f = x[:,:2048,:]
        x_m = x[:,2048:,:]
        x_f = self.to_tokens(x_f)
        x_m = self.to_mag(x_m)
        x_f = x_f+args.mag_ratio*x_m

        for backbone, conv in self.stages:
            x_f = backbone(x_f)
            if exists(conv):
                x_f = conv(x_f)

        x_f = x_f.permute(0, 2, 1)
        x =  self.to_logits(x_f)
        scores = self.sigmoid(self.fc(x))  # (B*10crop,32,1)
        score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores  = MSNSD(x,scores,bs,self.batch_size,self.drop_out,ncrops,k)

        return score_abnormal, score_normal, abn_feamagnitude, nor_feamagnitude, scores

