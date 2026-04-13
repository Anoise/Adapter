from tqdm import tqdm
from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, Buffer
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')
import torch.nn.functional as F

class PostProcessingNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, delta=0.1):
        super(PostProcessingNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.bn1 = torch.nn.InstanceNorm1d(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.InstanceNorm1d(hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.delta = delta
        print('PostProcessingNet delta:', self.delta)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)
        return torch.tanh(x) * self.delta + 1

class Exp_Decom(Exp_Basic):
    def __init__(self, args):
        super(Exp_Decom, self).__init__(args)
        self.opt = self._select_optimizer()
        
     
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=0.0000001)#self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        #criterion = nn.L1Loss()
        return criterion
    
    def _get_preds(self, batch_x, batch_y, batch_x_mark=None, batch_y_mark=None, i=None):

        if 'TST' in self.args.model:
            outputs = self.model(batch_x.permute(0,2,1), batch_x_mark, batch_y_mark)
            outputs = outputs.permute(0,2,1)
        elif "Period" in self.args.model or "Fre" in self.args.model:
            outputs = self.model(batch_x)
        elif 'iTrans' in self.args.model or 'DeepBooTS' in self.args.model:
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        elif 'former' in self.args.model or 'Linear' in self.args.model:
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            outputs = self.model(batch_x, dec_inp)
        elif 'rnn' in self.args.model:
            if len(batch_x.shape)<3:
                batch_x = batch_x.unsqueeze(0)
                batch_y = batch_y.unsqueeze(0)
            batch_x = batch_x.permute(0,2,1)
            batch_y = batch_y.permute(0,2,1)
            outputs = self.model(batch_x, batch_x_mark, batch_y_mark, batch_y)
        elif "Lade" in self.args.model or "Four" in self.args.model:
            outputs = self.model(batch_x)
        else:
            if len(batch_x.shape)<3:
                batch_x = batch_x.unsqueeze(0)
                batch_y = batch_y.unsqueeze(0)
            batch_x = batch_x.permute(0,2,1)
            batch_y = batch_y.permute(0,2,1)
            batch_x = batch_x.unsqueeze(-1)
            batch_y = batch_y.unsqueeze(-1)
            if self.args.model in ('STID','Mvstgn'):
                outputs = self.model(batch_x, batch_y_mark, batch_y_mark)
            else:
                outputs = self.model(batch_x, batch_y_mark, batch_y_mark, ycl=None, batch_seen=i)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        return outputs, batch_y    

    def vali(self, vali_loader, criterion):
        total_loss = []
        mae_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._get_preds(batch_x, batch_y, batch_x_mark, batch_y_mark, i)
                
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                mae = F.l1_loss(pred, true)
                mae_loss.append(mae)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        mae_loss = np.average(mae_loss)
        self.model.train()
        return total_loss, mae_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                self.opt.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)


                outputs, batch_y = self._get_preds(batch_x, batch_y, batch_x_mark, batch_y_mark, i)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.4f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.opt.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_mae_loss = self.vali(vali_loader, criterion)
            test_loss, test_mae_loss = self.vali(test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.4f} Vali Loss: {3:.4f} Val MAE Loss: {4:.4f} Test Loss: {5:.4f} Test MAE Loss: {6:.4f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, vali_mae_loss, test_loss, test_mae_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.opt, epoch + 1, self.args)

            # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
        

    def test(self, setting, test=1, visual_ts=False):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model y-mul .......')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        criterion = self._select_criterion()
        buffer = Buffer(self.args.seq_len, self.args.pred_len)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

            
            outputs, batch_y = self._get_preds(batch_x, batch_y, batch_x_mark, batch_y_mark, i)
            
            past_x, past_y =  buffer(batch_y)
            if past_x is not None:
                self.model.train()
                pred_past_y, _ = self._get_preds(past_x, past_y, None, None, i)
                loss = criterion(pred_past_y, past_y)
                loss.backward()
                self.opt.step()       
                self.opt.zero_grad()
                self.model.eval()
            
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            if test_data.scale and self.args.inverse:
                outputs = test_data.inverse_transform(outputs)
                batch_y = test_data.inverse_transform(batch_y)

            pred = outputs
            true = batch_y

            preds.append(pred)
            trues.append(true)

            if visual_ts and i % 20 == 0:
                input = batch_x.detach().cpu().numpy()
                gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{0:4f}, mae:{1:4f}, rmse:{2:4f}, mape:{3:4f}, mspe:{4:4f},'.format(mse, mae, rmse, mape, mspe))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + " \n")
        f.write('mse:{0:4f}, mae:{1:4f}, rmse:{2:4f}, mape:{3:4f}, mspe:{4:4f},'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return


    def continue_train(self, setting, vali_set = False):
        print('continue training the model....')
        print("load model ...")
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        continue_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.continue_train_lr)
        
        if vali_set:
            _, data_loader = data_provider(self.args, 'val')
        else:
            _, data_loader = data_provider(self.args, 'train')

        _, test_loader = data_provider(self.args, 'test')
        criterion = torch.nn.MSELoss()
        
        for episode in range(self.args.num_episodes):
            loss_list = []
            with tqdm(total=len(data_loader), desc='Iteration %d' % episode) as pbar:
                for step, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    outputs, batch_y = self._get_preds(batch_x, batch_y, batch_x_mark, batch_y_mark, episode)

                    loss = criterion(outputs, batch_y)
                    loss_list.append(loss.item())
                    loss.backward()
                    continue_optim.step()

                    if (step+1) % 10 == 0:
                        pbar.set_postfix({'step': '%d' % (step+1), 'loss': '%.4f' % np.mean(loss_list)})
                    pbar.update(1)
            
            path = os.path.join(self.args.checkpoints, setting)
            test_loss, mae_loss = self.vali(test_loader, criterion)
            print('test_loss=%.4f' % test_loss, 'mae_loss=%.4f' % mae_loss)

        torch.save(self.model.state_dict(), path + '/' + 'continue_ckpt_{}.pth'.format(vali_set))
        
        return test_loss

    def vali_post(self, post_net, vali_loader, criterion):
        total_loss = []
        mae_loss = []
        self.model.eval()
        post_net.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # x_state = batch_x.view(batch_x.shape[0],-1)
                # infos = post_net(x_state)
                # batch_x = batch_x * infos.view(*batch_x.shape) 
                
                outputs, batch_y = self._get_preds(batch_x, batch_y, batch_x_mark, batch_y_mark, i)
                
                action = post_net(outputs.reshape(outputs.shape[0],-1))
                outputs = outputs * action.view(*outputs.shape) 
                
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                mae = F.l1_loss(pred, true)
                mae_loss.append(mae)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        mae_loss = np.average(mae_loss)
        post_net.train()
        return total_loss, mae_loss
   
    def post_train(self, setting, vali_set = False):
        
        print("load model ...")
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        self.model.eval()
        
        post_net = PostProcessingNet(self.args.seq_len * self.args.enc_in, 
                                         self.args.d_model, 
                                         self.args.pred_len * self.args.enc_in,
                                         self.args.delta).to(self.device)
        self.post_optim = torch.optim.Adam(post_net.parameters(), lr=self.args.post_train_lr)
        
        if vali_set:
            _, data_loader = data_provider(self.args, 'val')
        else:
            _, val_loader = data_provider(self.args, 'val')
            _, data_loader = data_provider(self.args, 'train')
            num_episodes = self.args.num_episodes

        _, test_loader = data_provider(self.args, 'test')
        criterion = torch.nn.MSELoss()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        for episode in range(self.args.num_episodes):
            loss_list = []
            with tqdm(total=len(data_loader), desc='Iteration %d' % episode) as pbar:
                for step, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                    # x_state = batch_x.view(batch_x.shape[0],-1)
                    # infos = post_net(x_state)
                    # batch_x = batch_x * infos.view(*batch_x.shape) 
                    
                    outputs, batch_y = self._get_preds(batch_x, batch_y, batch_x_mark, batch_y_mark, episode)
                    
                    y_state = outputs.reshape(outputs.shape[0],-1)
                    action = post_net(y_state)
                    outputs = outputs * action.view(*batch_y.shape)
                    
                    loss = criterion(outputs, batch_y) 
                    loss_list.append(loss.item())
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(post_net.parameters(), 0.1)
                    self.post_optim.step()

                    if (step+1) % 10 == 0:
                        pbar.set_postfix({'step': '%d' % (step+1), 'loss': '%.4f' % np.mean(loss_list)})
                    pbar.update(1)
                
            if not vali_set:
                val_loss, mae_loss = self.vali_post(post_net, val_loader, criterion)
                print('val_loss=%.4f' % val_loss, 'mae_loss=%.4f' % mae_loss)
                
                path = os.path.join(self.args.checkpoints, setting) 
                early_stopping(val_loss, self.model, path, name='post_ckpt_{}.pth'.format(vali_set))
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                    
            test_loss, mae_loss = self.vali_post(post_net, test_loader, criterion)
            print('test_loss=%.4f' % test_loss, 'mae_loss=%.4f' % mae_loss)
            
            adjust_learning_rate(self.post_optim, episode + 1, self.args)
            
        # path = os.path.join(self.args.checkpoints, setting)
        # torch.save(post_net.state_dict(), path + '/' + 'post_ckpt_{}.pth'.format(vali_set))
        
        return test_loss



    def test2(self, setting, continue_train=False, post_process=False, vali_set=False, post_noise = False, visual_ts=False):
        ''' 继续训练 与 后处理阶段的测试 '''
        test_data, test_loader = self._get_data(flag='test')
        if continue_train:
            print('loading continue_ckpt_{}.pth ...'.format(vali_set))
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 
                                                               'continue_ckpt_{}.pth'.format(vali_set))))
            self.model.eval()
        else:
            print('loading checkpoint.pth ...')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            self.model.eval()

        if post_process:
            print('post_ckpt_{}.pth ...'.format(vali_set))
            post_net = PostProcessingNet(self.args.seq_len * self.args.enc_in, 
                                         self.args.d_model, self.args.seq_len * self.args.enc_in).to(self.device)
            post_net.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'post_ckpt_{}.pth'.format(vali_set))))
            
            self.post_optim = torch.optim.Adam(post_net.parameters(), lr=0.0001)
          

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        criterion = self._select_criterion()
        buffer = Buffer(self.args.seq_len, self.args.pred_len)
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            
            # ### 后处理               
            # if post_process:
            #     x_state = batch_x.view(batch_x.shape[0],-1)
            #     infos = post_net(x_state)
            #     batch_x = batch_x *  infos.view(*batch_x.shape)
            if post_noise:
                infos = torch.randn_like(batch_x)/20
                batch_x = batch_x + infos

            if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
            
            outputs, batch_y = self._get_preds(batch_x, batch_y, batch_x_mark, batch_y_mark, i)
            
            ### 后处理               
            if post_process:
                y_state = outputs.reshape(outputs.shape[0],-1)
                action = post_net(y_state)
                outputs = outputs * action.view(*batch_y.shape)
            
            past_x, past_y =  buffer(batch_y)
            if past_x is not None and post_process:
                # past_state = past_x.view(past_x.shape[0],-1)
                # infos = post_net(past_state)
                # past_x = past_x * infos.view(*past_x.shape)
                
                pred_past_y, _ = self._get_preds(past_x, past_y, None, None, i)
                
                y_state = pred_past_y.reshape(pred_past_y.shape[0],-1)
                action = post_net(y_state)
                pred_past_y = pred_past_y * action.view(*past_y.shape)
                
                loss = criterion(pred_past_y, past_y)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(post_net.parameters(), 0.1)
                self.post_optim.step()       
                self.post_optim.zero_grad()

            
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            if test_data.scale and self.args.inverse:
                outputs = test_data.inverse_transform(outputs)
                batch_y = test_data.inverse_transform(batch_y)

            pred = outputs
            true = batch_y
            
            preds.append(pred)
            trues.append(true)
            if visual_ts and i % 20 == 0:
                _input = batch_x.detach().cpu().numpy()
                gt = np.concatenate((_input[0, :, -1], true[0, :, -1]), axis=0)
                pd = np.concatenate((_input[0, :, -1], pred[0, :, -1]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{0:4f}, mae:{1:4f}, rmse:{2:4f}, mape:{3:4f}, mspe:{4:4f},'.format(mse, mae, rmse, mape, mspe))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + ", continue_train={}, post_process={}, vali_set={}, post_noise={}. \n".format(continue_train, post_process, vali_set, post_noise))
        f.write('mse:{0:4f}, mae:{1:4f}, rmse:{2:4f}, mape:{3:4f}, mspe:{4:4f},'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return mse
   
    
