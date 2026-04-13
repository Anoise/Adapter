import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Minusformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer

class standard_scaler_v2():
    def __init__(self, ts, dim=-1):
        self.mean = ts.mean(dim, keepdim=True)
        self.std = torch.sqrt(torch.var(ts-self.mean, dim=dim, keepdim=True, unbiased=False) + 1e-5)

    def transform(self, data):
        data = (data - self.mean) / self.std
        return data

    def inverted(self, data):
        data = (data * self.std) + self.mean
        return data

class DecomModel(nn.Module):
    def __init__(self, configs):
        super(DecomModel, self).__init__()
        self.pred_len = configs.pred_len
        self.embed = nn.Linear(configs.seq_len, configs.d_model)
        self.backbone = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads) if configs.attn else None,
                    configs.d_model,
                    configs.d_block,
                    configs.d_ff,
                    dropout=configs.dropout,
                    gate = configs.gate
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        if configs.d_block != configs.pred_len:
            self.align = nn.Linear(configs.d_block, configs.pred_len)
        else:
            self.align = nn.Identity()
            print('Identity ...')

        print('Minusformer ...')

    def forward(self, x):
        x_emb = self.embed(x)
        output = self.backbone(x_emb)
        return output[:, :x.size(1), :]
 


class StatiModel(torch.nn.Module):
    def __init__(self, configs):
        super(StatiModel, self).__init__()
        out_channels= configs.d_model * 2
        self.pred_len = configs.pred_len
        
        self.l_m1 = torch.nn.Linear(configs.seq_len, out_channels//4)
        self.l_m2 = torch.nn.Linear(out_channels//4, configs.pred_len)
        
        self.l_s1 = torch.nn.Linear(configs.seq_len, out_channels//4)
        self.l_s2 = torch.nn.Linear(out_channels//4, configs.pred_len)

        print("Statistics Model ...")
    
    def forward(self, x_m):
        
        m1 = self.l_m1(x_m)
        m1 = F.leaky_relu(m1)
        m2 = self.l_m2(m1)

        s1 = self.l_s1(x_m)
        s1 = F.leaky_relu(s1)
        s2 = self.l_s2(s1)
        
        return m2, s2



class Model(torch.nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.stati_model = StatiModel(configs)
        self.decom_model = DecomModel(configs)
        
        print("Lade Model ...")
        
    def forward(self, x, y=None, criterion=None, edge_indexs=None, edge_weights=None):
        
        x = x.permute(0,2,1)
        ### decomposition of x
        scaler = standard_scaler_v2(x)
        _x = scaler.transform(x)
        
        mean, std = self.stati_model(x)
        
        if y is not None:
            y_scaler = standard_scaler_v2(y.permute(0,2,1))
            out = self.decom_model(_x)
            out = out * std + mean
            stati_loss =  criterion(mean, y_scaler.mean) + criterion(std, y_scaler.std)
            
            return out.permute(0,2,1), stati_loss
        
            # mean = scaler.mean+mean
            # std = scaler.std+std
        out = self.decom_model(_x)
        out = out * std + mean

        # if y is not None:
        #     assert criterion is not None
        #     # _y = scaler.transform(y.permute(0,2,1))
        #     # stati_loss =  criterion(mean, _y.mean(-1,keepdim=True)) + criterion(std, _y.std(-1, keepdim=True))
            

        return out.permute(0,2,1)