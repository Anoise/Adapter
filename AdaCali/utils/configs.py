import argparse
import torch

from model import Reformer, Flowformer, Flashformer, \
    iTransformer, DeepBooTS, Transformer

from models_offline import Autoformer, Informer, PatchTST, FEDformer, \
      DLinear, Periodformer, FreTS, FourierGNN

model_dict = {
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
    'DeepBooTS':DeepBooTS
        }


class Configs():
    def __init__(self, data='ELC', root_path='$your_data_path'):
            
        # basic config
        self.is_training=1
        self.model_id='test'
        self.model='iTransformer'

        # data loader
        self.data='ELC'
        self.root_path='$your_data_path'
        self.data_path='electricity.csv'
        self.features='S'
        self.target='OT'
        self.freq='h'
        self.checkpoints='./checkpoints/'
        self.distil=True
        self.seasonal_patterns = 'Monthly'

        # forecasting task
        self.seq_len=96#='input sequence length')
        self.label_len=48#='start token length') # no longer needed in inverted Transformers
        self.pred_len=96#='prediction sequence length')

        # model define
        self.d_block=None
        self.d_model=512
        self.n_heads=8
        self.e_layers=2
        self.d_layers=1
        self.d_ff=2048
        self.moving_avg=25
        self.factor=1
        self.attn=1
        self.gate=1

 
        self.dropout=0.1
        self.embed='timeF'
        self.activation='gelu'
        self.output_attention=False

        # optimization
        self.num_workers=10
        self.itr=1
        self.train_epochs=10
        self.batch_size=32#='batch size of train input data')
        self.patience=3#='early stopping patience')
        self.learning_rate=0.0001#='optimizer learning rate')
        self.des='test'#='exp description')
        self.loss='MSE'#='loss function')
        self.lradj='type1'#='adjust learning rate')

        # GPU
        self.use_gpu=True#='use gpu')
        self.gpu=0#='gpu')
        self.use_multi_gpu=False
        self.devices='0,1,2,3'#='device ids of multile gpus')

        # iTransformer
        self.exp_name='None'
        self.efficient_training=False#='whether to use efficient_training (exp_name should be partial train)')
        self.channel_independence=False#='whether to use channel_independence mechanism')
        self.inverse=False
        self.class_strategy='projection'#='projection/average/cls_token')
        self.target_root_path='./data/electricity/'#='root path of the data file')
        self.target_data_path='electricity.csv'#='data file')


        ### RL
        self.post_train_lr=1e-4#='optimizer learning rate')
        self.continue_train_lr=1e-7#='optimizer learning rate')    
        self.num_episodes=10#='number of episodes')
        self.gamma=0.9#='gamma of the reward of PPO')
        self.lmbda=0.9#='lmbda of the reword of PPO')
        self.eps=0.2#='clip rate')

        self.use_gpu = True if torch.cuda.is_available() and self.use_gpu else False

        if self.use_gpu and self.use_multi_gpu:
            self.devices = self.devices.replace(' ', '')
            device_ids = self.devices.split(',')
            self.device_ids = [int(id_) for id_ in device_ids]
            self.gpu = self.device_ids[0]

        self.data = data
        self.root_path = root_path
        self.model_id=data+'_96_'+str(self.pred_len)

        data_parser = {
        'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
        'Weather':{'data':'weather.csv','T':'OT','M':[21,21,21],'S':[1,1,1],'MS':[21,21,1]},
        'ELC':{'data':'electricity.csv','T':'OT','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
        'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
        'Toy': {'data': 'Toy.csv', 'T':'Value', 'S':[1,1,1]},
        'ToyG': {'data': 'ToyG.csv', 'T':'Value', 'S':[1,1,1]},
        'Exchange': {'data': 'exchange_rate.csv', 'T':'OT', 'M':[8,8,8], 'S':[1,1,1]},
        'Illness': {'data': 'national_illness.csv', 'T':'OT', 'M':[7,7,7], 'S':[1,1,1]},
        'Traffic': {'data': 'traffic.csv', 'T':'OT', 'M':[862,862,862], 'S':[1,1,1]},
        }
        if self.data in data_parser.keys():
            data_info = data_parser[self.data]
            self.data_path = data_info['data']
            self.target = data_info['T']
            self.enc_in, self.dec_in, self.c_out = data_info[self.features]
            self.data='custom'


        # setting record of experiments
        self.setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            self.model_id,
            self.model,
            self.data,
            self.features,
            self.seq_len,
            self.label_len,
            self.pred_len,
            self.d_model,
            self.n_heads,
            self.e_layers,
            self.d_layers,
            self.d_ff,
            self.factor,
            self.embed,
            self.distil,
            self.des,
            self.class_strategy, 0)


