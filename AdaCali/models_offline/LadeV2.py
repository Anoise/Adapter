import torch
import torch.nn.functional as F


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
        m1 = F.relu(m1)
        m2 = self.l_m2(m1)

        s1 = self.l_s1(x_m)
        s1 = F.relu(s1)
        s2 = self.l_s2(s1)
        
        return m2, s2


class DecomModel(torch.nn.Module):
    def __init__(self, configs):
        super(DecomModel, self).__init__()
        out_channels= configs.d_model * 2
        self.pred_len = configs.pred_len
        
        self.l_x1 = torch.nn.Linear(configs.seq_len, out_channels)
        self.l_x2 = torch.nn.Linear(out_channels, configs.pred_len)
        
        self.l_o1 = torch.nn.Linear(configs.pred_len, out_channels)
        self.l_o2 = torch.nn.Linear(out_channels, configs.pred_len)

        print("Decomposition Model ...")

    def forward(self, x, mean, std, edge_indexs=None, edge_weights=None):
        
        x1 = self.l_x1(x)
        x1 = F.relu(x1)
        x2 = self.l_x2(x1)

        out = x2 * std

        out = self.l_o1(out)
        out = F.relu(out)
        out = self.l_o2(out) + mean

        return out
    


class Model(torch.nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.stati_model = StatiModel(configs)
        self.decom_model = DecomModel(configs)
        
        print("Lade V2 Model ...")
        
    def forward(self, x, y=None, criterion=None, edge_indexs=None, edge_weights=None):
        
        x = x.permute(0,2,1)
        ### decomposition of x
        scaler = standard_scaler_v2(x)
        _x = scaler.transform(x)
        
        mean, std = self.stati_model(x)
        y_scaler = standard_scaler_v2(y.permute(0,2,1))
        # if y is not None:
            
        #     out = self.decom_model(_x, y_scaler.mean, y_scaler.std)
        #     stati_loss =  criterion(mean, y_scaler.mean) + criterion(std, y_scaler.std)
            
        #     return out.permute(0,2,1), stati_loss
        
            # mean = scaler.mean+mean
            # std = scaler.std+std
        out = self.decom_model(_x, y_scaler.mean, y_scaler.std)

        # if y is not None:
        #     assert criterion is not None
        #     # _y = scaler.transform(y.permute(0,2,1))
        #     # stati_loss =  criterion(mean, _y.mean(-1,keepdim=True)) + criterion(std, _y.std(-1, keepdim=True))
            

        return out.permute(0,2,1)
        
    def decompose(self, inputs):
        means = inputs.mean(-1, keepdim=True).detach()
        inputs = inputs - means
        stdev = torch.sqrt(torch.var(inputs, dim=-1, keepdim=True, unbiased=False) + 1e-7)
        inputs /= stdev
        return inputs, means, stdev