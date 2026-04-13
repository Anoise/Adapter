import argparse
import torch
from experiments.exp_quan import Exp_Decom
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='RL Predictor')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='DecomLinear', help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./iTransformer_datasets/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--d_block', type=int, help='dimension of model')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--attn', type=int, default=1, help='use attention (1) or not (0)')
    parser.add_argument('--gate', type=int, default=1, help='use gate (1) or not (0)')
    
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--exp_name', type=str, required=False, default='None',
                        help='experiemnt name, options:[partial_train, zero_shot]')
    parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')


    ### RL
    parser.add_argument('--post_train_lr', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--continue_train_lr', type=float, default=1e-7, help='optimizer learning rate')    
    parser.add_argument('--num_episodes', type=int, default=10, help='number of episodes')
    parser.add_argument('--gamma', type=float, default=0.9, help='gamma of the reward of PPO')
    parser.add_argument('--lmbda', type=float, default=0.9, help='lmbda of the reword of PPO')
    parser.add_argument('--eps', type=float, default=0.2, help='clip rate')
    
    # PatchTST
    parser.add_argument('--patch_len', type=int, default=2, help='patch length')
    parser.add_argument('--stride', type=int, default=2, help='stride')
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    
    parser.add_argument('--version', type=str, default='Fourier',
                    help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')


    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.d_block is None: args.d_block = args.pred_len

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.model =="Periodformer":
        args.period = args.seq_len // 4

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
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]
        args.data='custom'

    print('Args in experiment:')
    print(args)

    Exp = Exp_Decom

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                args.class_strategy, ii)

            exp = Exp(args)  # set experiments

            # ### ------------- 训练网络 --------------------
            # print('\n>>>>>>> start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            # exp.train(setting)

            # print('\n>>>>>>> just testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test(setting)
  

            #taus = torch.tensor([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975], dtype=torch.float32) 
            taus = torch.tensor([0.025, 0.975], dtype=torch.float32) 
            #exp.train_quantile_calibrator(setting, taus, max_epochs=5)
            exp.test_quantile_calibrator(setting, taus)
            exit()


            # ## ------------- 噪声测试 --------------------
            
            # # ## 直接添加噪声，进行测试
            # print('\n>>>>>>> postprocessing with noise : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test2(setting, post_noise=True)
            
            # # ### ------------- 继续训练 --------------------
            
            # # # 使用 训练集 进行继续训练
            # print('\n<<<<<<< continue training with trianset: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            # exp.continue_train(setting)
            
            # print('\n>>>>>>> continue test : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test2(setting, continue_train=True)
            
            # # # # 使用 测试集进行继续训练
            
            # print('\n<<<<<<< continue training with vali-set: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            # exp.continue_train(setting, vali_set = True)
            
            # print('\n>>>>>>> continue test with vali-set: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test2(setting, continue_train=True, vali_set=True)
            
            ### ------------- 后处理网络 --------------------
            # 使用 训练集 重新训练网络
            # print('\n<<<<<<< postprocessing with trainset : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            # exp.post_train(setting)
            
            # print('\n>>>>>>> postprocessing test with trainset : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            # exp.test2(setting, post_process=True, vali_set=False)
            #exit()
            ### 使用 验证集 重新训练网络

            # print('\n<<<<<<< postprocessing with vali_set : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            # exp.post_train(setting, vali_set=True)

            # print('\n>>>>>>> postprocessing test with vali_set : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            # exp.test2(setting, post_process=True, vali_set=True)
            # exit()
            ### ------------- 继续训练 + 后处理网络 --------------------
            # 继续训练 + 后处理 
            # print('\n>>>>>>> C+P test with trainset : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            # exp.test2(setting, continue_train=True, post_process=True)
            
            # # 继续训练 + 后处理 + 测试集

            # print('\n>>>>>>> C+P test with vali_set : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            # exp.test2(setting, continue_train=True, post_process=True, vali_set = True)
            
            # # 继续训练 + 后处理 + 测试集 + 噪声
            # print('\n>>>>>>> C+P test with vali_set and noise : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            # exp.test2(setting, continue_train=True, post_process=True, vali_set = True, post_noise = True)
            
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            args.class_strategy, ii)

        exp = Exp(args)  # set experiments
        print('\n>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        # exp.train_on_policy_agent(setting)
        # exp.test2(setting, test=1)
        torch.cuda.empty_cache()
