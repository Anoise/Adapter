import os
import torch
from model import Reformer, Flowformer, Flashformer, \
    iTransformer, DeepBooTS, Transformer
    
from models_offline import Autoformer, Informer, PatchTST, FEDformer, \
      Mvstgn, DLinear, Periodformer, PSLD, FreTS, FourierGNN, LadeV2, DeepBooTS_V2

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Transformer': Transformer,
            'Autoformer': Autoformer,
            'PatchTST': PatchTST,
            'FEDformer': FEDformer,
            'DLinear': DLinear,
            'Periodformer': Periodformer,
            'FreTS': FreTS,
            'FourierGNN': FourierGNN,
            'Informer': Informer,
            'Reformer': Reformer,
            'Flowformer': Flowformer,
            'Flashformer': Flashformer,
            'iTransformer': iTransformer,
            'DeepBooTS':DeepBooTS,
            'LadeV2':LadeV2,
            'DeepBooTS_V2':DeepBooTS_V2,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
