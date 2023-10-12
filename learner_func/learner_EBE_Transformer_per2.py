import time
import pandas as pd
import numpy as np
from meta_models import EEG_EBE_Transformer_per2
from utils.data_training_EBE_transformer import train_epoch_enhanceTrain_chsz, train_epoch_enhanceTrain_tusz, val_tusz,val_chsz
from torch import nn
from torch.utils.data import DataLoader
from my_config import Config
import os
import sklearn.metrics as sm
from utils.pytorchtools import EarlyStopping
from torch.utils.data.sampler import WeightedRandomSampler
import os.path
import joblib
from utils.lib import print_time_cost, vote

import my_config
import torch, gc

from utils.test_utils import output, output_waves_datt

GPU_ID1 = my_config.Config.GPU_id1
os.environ['CUDA_VISIBLE_DEVICES']=str(GPU_ID1)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

torch.manual_seed(1043)
torch.cuda.manual_seed(1043)
default_alpha=0.8
default_temp = 5
import matplotlib.pyplot as plt
import argparse
#KD_loss: sharpKD/KL/JSD
def trainer_data_augmentation_TimeTransformer_base(myran, patch_size, run_time=5, folds=5, mmd= False,
                                                   pearson = False,l=3,mybeta=0.01,mylambda=0.01,
                                                   weight_Type = 'Avg',is_L1=True,model_name_base = 'WHC',
                                                   alpha=1, temp=6):
    if model_name_base == 'WHC':
        load_path = my_config.Config.WHC_data
        from DataLoaders.CHSZ_EEG_Dataloader import Patient_allocate, WHC_dataset, BuildTestData
        data_eeg, delta_eeg, theta_eeg, alpha_eeg, beta_eeg, gamma_eeg, upper_eeg, label, pid_list = BuildTestData.testdata_all(load_path)
        label_mark, groupnum, patients_id = Patient_allocate(pid_list, label, folds=folds, ran = myran)
        pid = pd.DataFrame(np.array(patients_id).reshape(-1, 1), columns=['pid'])
        gno = pd.DataFrame(np.array(groupnum).reshape(-1, 1), columns=['gno'])
        combine = pd.concat((pid, gno), axis=1)

    else:
        load_path = my_config.Config.TUSZ_data
        model_name_base = 'TUSZ'
        from DataLoaders.TUSZ_EEG_Dataloader import Patient_allocate, TUSZ_dataset, BuildTestData
        data_eeg, delta_eeg, theta_eeg, alpha_eeg, beta_eeg, gamma_eeg, upper_eeg, label, pid_list = BuildTestData.testdata_all(load_path)
        label_mark, groupnum, patients_id = Patient_allocate(pid_list, label, folds=folds, ran = myran)
        pid = pd.DataFrame(np.array(patients_id).reshape(-1, 1), columns=['pid'])
        gno = pd.DataFrame(np.array(groupnum).reshape(-1, 1), columns=['gno'])
        combine = pd.concat((pid, gno), axis=1)

    acc_list_p1 = []
    bca_list_p1 = []
    f1_list_p1 = []

    for run in range(run_time):
        acc_all_p1 = 0
        bca_all_p1 = 0
        f1_all_p1 = 0
        ACC_p1 = np.zeros(folds)
        BCA_p1 = np.zeros(folds)
        F1_p1 = np.zeros(folds)

        num_events = np.zeros(folds)
        Test_label = []

        for i in range(folds):

            test_pid = combine[combine['gno'] == i]['pid']
            print(test_pid)
            test_pid_list = test_pid.tolist()

            train_pid = combine[combine['gno'] != i]['pid']
            print(train_pid)
            train_pid_list = train_pid.tolist()

            if model_name_base == 'WHC':
                train_dataset = WHC_dataset(path=load_path, test_pid=test_pid_list, train=True, overlap=True)
                val_dataset_o = WHC_dataset(path=load_path, test_pid=test_pid_list, train=False, overlap=True)
                train_nums = len(train_dataset.eeg_data)
            else:
                train_dataset = TUSZ_dataset(path=load_path, test_pid=test_pid_list, train=True, overlap=True)
                val_dataset_o = TUSZ_dataset(path=load_path, test_pid=test_pid_list, train=False, overlap=True)
                train_nums = len(train_dataset.eeg_data)

            sweight_train = torch.tensor(train_dataset.sample_weight)
            sampler_t = WeightedRandomSampler(sweight_train, num_samples=train_nums, replacement=True)
            train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=False, sampler=sampler_t, drop_last=True, num_workers=4)
            val_dataloader = DataLoader(val_dataset_o, batch_size=Config.batch_size, shuffle=False, num_workers=2)

            print("train_datasize", len(train_dataset), "val_o_datasize", len(val_dataset_o))
            # train
            best_f1 = -1
            lr = Config.lr
            start_epoch = 1
            stage = 1

            model_A = EEG_EBE_Transformer_per2.Multi_TimeTransformer(patch_size=patch_size)
            model_A = model_A.to(device)

            model_name_A = model_name_base + '_EEG_EBE_Transformer_' \
                           + '_ran_' + str(myran) +'_L1_'+str(is_L1)+'_beta_'+str(mybeta) \
                           + '_pearson_' + str(pearson) + '_mmd_' + str(mmd) \
                           + '_per2_ML_' + str(mylambda)+ '_l_' + str(l) \
                           +'_temp_'+str(temp)+ '_KFold_' + str(i) \
                           + '_runtime_' + str(run) + 'fold_' + str(folds)
            # model_A = model_A.to(device)
            save_path_A = os.path.join(Config.model_save_path_multi_temp + 'Weight_Type_EBE_ML/'+ 'weight_type_'+str(weight_Type)+'/', model_name_A)

            if not os.path.exists(save_path_A):
                os.makedirs(save_path_A, exist_ok=True)

                optimizerA = torch.optim.AdamW(model_A.parameters(), lr=Config.lr, betas=(0.9, 0.999),
                                               weight_decay=5e-5)


                criterion_p = nn.CrossEntropyLoss()  # KL/sharpKD

                early_stopping_p = EarlyStopping(patience=Config.patience, verbose=True,
                                                 path=os.path.join(save_path_A, 'checkpoint.pt'))

                epc = 0
                train_ls_A, val_ls_A = [], []
                train_ls_B, val_ls_B = [], []
                best_epoch = Config.max_epoch

                if model_name_base == 'WHC':
                    for epoch in range(start_epoch, Config.max_epoch + 1):
                        epc += 1
                        since = time.time()
                        train_loss_A, train_f1_A= train_epoch_enhanceTrain_chsz(model=model_A,
                                                                                optimizer=optimizerA,
                                                                                criterion = criterion_p,
                                                                                train_dataloader=train_dataloader,
                                                                                show_interval=100,
                                                                                if_MMD= mmd,
                                                                                weight_Type= weight_Type,
                                                                                if_pearson= pearson,
                                                                                device=device,
                                                                                reshape_flag=False,
                                                                                lamda=mylambda,
                                                                                alpha=alpha,
                                                                                l=l,
                                                                                is_L1 = is_L1,
                                                                                beta=mybeta,
                                                                                temp=temp)

                        val_output_A, val_loss_A, val_f1_A, val_bca_A \
                            = val_chsz(model=model_A, criterion=criterion_p,
                                          val_dataloader=val_dataloader, device=device, reshape_flag=False)

                        train_ls_A.append(train_loss_A)
                        val_ls_A.append(val_loss_A)

                        print('#Model 1: epoch:%02d stage:%d train_loss:%.3e val_loss:%0.3e val_f1:%.3f val_bca:%.3f time:%s\n'
                              % (epoch, stage, train_loss_A, val_loss_A, val_f1_A, val_bca_A, print_time_cost(since)))

                        early_stopping_p(val_loss_A, model_A)

                        if early_stopping_p.early_stop:
                            print("Early stopping")
                            best_epoch = epoch
                            print('Best epoch is: ', best_epoch)
                            break
                    print("Model 1: final epoch: train loss %f, test loss %f" % (train_ls_A[-1], val_ls_A[-1]))

                else:
                    for epoch in range(start_epoch, Config.max_epoch + 1):
                        epc += 1
                        since = time.time()
                        train_loss_A, train_f1_A= train_epoch_enhanceTrain_tusz(model=model_A,
                                                                                optimizer=optimizerA,
                                                                                criterion = criterion_p,
                                                                                train_dataloader=train_dataloader,
                                                                                show_interval=100,
                                                                                if_MMD= mmd,
                                                                                weight_Type= weight_Type,
                                                                                if_pearson= pearson,
                                                                                device=device,
                                                                                reshape_flag=False,
                                                                                lamda=mylambda,
                                                                                alpha=alpha,
                                                                                l=l,
                                                                                is_L1 = is_L1,
                                                                                beta=mybeta,
                                                                                temp=temp)
                        val_output_A, val_loss_A, val_f1_A, val_bca_A \
                            = val_tusz(model=model_A, criterion=criterion_p,
                            val_dataloader=val_dataloader, device=device, reshape_flag=False)

                        train_ls_A.append(train_loss_A)
                        val_ls_A.append(val_loss_A)

                        print(
                            '#Model 1: epoch:%02d stage:%d train_loss:%.3e val_loss:%0.3e val_f1:%.3f val_bca:%.3f time:%s\n'
                            % (epoch, stage, train_loss_A, val_loss_A, val_f1_A, val_bca_A, print_time_cost(since)))

                        early_stopping_p(val_loss_A, model_A)

                        if early_stopping_p.early_stop:
                            print("Early stopping")
                            best_epoch = epoch
                            print('Best epoch is: ', best_epoch)
                            break
                    print("Model 1: final epoch: train loss %f, test loss %f" % (train_ls_A[-1], val_ls_A[-1]))

            print('-----test------')
            # gc.collect()
            # torch.cuda.empty_cache()

            data_eeg, delta_eeg, theta_eeg, alpha_eeg, beta_eeg, gamma_eeg, upper_eeg, label, pid_list = BuildTestData.testdata_all(load_path)
            student_my_data = data_eeg

            Amodel_path = save_path_A
            model_A = EEG_EBE_Transformer_per2.Multi_TimeTransformer(patch_size=patch_size)
            model_A.load_state_dict(torch.load(os.path.join(Amodel_path, 'checkpoint.pt'), map_location=device))
            model_A.to(device).eval()

            # 测试各个子模型和联合模型的结果
            target_label = []

            trail_pred_label_peer1 = []
            event_pred_label_peer1 = []

            for test_pid in test_pid_list:
                event_times = pid_list.count(test_pid)
                for tim in range(event_times):
                    test_dataloader = BuildTestData.testbuild(test_pid, tim, student_my_data, label, pid_list)
                    event_pred, test_label = output_waves_datt(model_A, test_dataloader, device)
                    target_label.append(test_label)
                    event_pred_label_peer1.append(event_pred)

            target_label_flatten = np.array(target_label).flatten()
            target_label = np.array(target_label_flatten)

            event_label_flatten_peer1 = [x for tup in event_pred_label_peer1 for x in tup]
            event_pred_label_peer1 = np.array(event_label_flatten_peer1)

            acc_p1 = sm.accuracy_score(target_label, event_pred_label_peer1)

            print('current  acc: ', acc_p1)
            acc_all_p1 += acc_p1

            bca_p1 = sm.balanced_accuracy_score(target_label, event_pred_label_peer1)

            print('current bca: ', bca_p1)
            bca_all_p1 += bca_p1

            f1_p1 = sm.f1_score(target_label, event_pred_label_peer1, average='weighted')

            print('current f1: ', f1_p1)
            f1_all_p1 += f1_p1

            ACC_p1[i] = acc_p1
            BCA_p1[i] = bca_p1
            F1_p1[i] = f1_p1

            num_events[i] = len(target_label)
            Test_label.extend(target_label.tolist())
            gc.collect()
            torch.cuda.empty_cache()

        output_path = 'output/EBE_Transformer_ML/weight_type_'+str(weight_Type)+'/if_L1_'+str(is_L1)+'_beta_'+str(mybeta)+'/'
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        joblib.dump({'ACC_ViTlow': ACC_p1, 'BCA_ViTlow': BCA_p1, 'F1_ViTlow': F1_p1,
                     'num_events': num_events, 'test_label': Test_label},
                    'output/EBE_Transformer_ML/weight_type_{}/if_L1_{}_beta_{}/{}_per2_EBE_ML_{}_mmd_{}_pearson_{}_temp_{}_l_{}_folds_{}_alpha_{}_run_time_{}.joblib'
                    .format(weight_Type,is_L1, mybeta, model_name_base,  mylambda ,mmd, pearson, temp, l, folds, alpha, run))
        acc_list_p1.append(acc_all_p1 / folds)
        bca_list_p1.append(bca_all_p1 / folds)
        f1_list_p1.append(f1_all_p1 / folds)

    acc_times_avg_p1 = np.average(np.array(acc_list_p1))
    bca_times_avg_p1 = np.average(np.array(bca_list_p1))
    f1_times_avg_p1 = np.average(np.array(f1_list_p1))
    acc_times_var_p1 = np.var(np.array(acc_list_p1))
    bca_times_var_p1 = np.var(np.array(bca_list_p1))
    f1_times_var_p1 = np.var(np.array(f1_list_p1))

    print(acc_times_avg_p1, bca_times_avg_p1, f1_times_avg_p1)
    print(acc_times_var_p1, bca_times_var_p1, f1_times_var_p1)

    joblib.dump({'ACC_avg_p1': acc_times_avg_p1, 'BCA_avg_p1': bca_times_avg_p1, 'F1_avg_p1': f1_times_avg_p1,
                 'ACC_var_p1': acc_times_var_p1, 'BCA_var_p1': bca_times_var_p1, 'F1_var_p1': f1_times_var_p1,
                 'folds': folds},
                'output/EBE_Transformer_ML/weight_type_{}/if_L1_{}_beta_{}/{}_per2_EBE_ML_{}_mmd_{}_pearson_{}_temp_{}_l_{}_alpha_{}_pat_{}.joblib'
                .format(weight_Type,is_L1, mybeta, model_name_base, mylambda, mmd, pearson, temp, l, alpha, myran))



