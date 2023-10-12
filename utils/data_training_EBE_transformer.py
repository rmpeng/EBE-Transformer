import torch
from utils.KD_loss import loss_KL, JSD_loss
from utils.MMD_distance import MMD_loss
from torch.nn import functional as F
from my_config import Config
import sklearn.metrics as sm
from audtorch.metrics.functional import pearsonr

def old_reshape_input(input):
    input = input.permute(0, 2, 1, 3)
    return input

def train_epoch_enhanceTrain_chsz(model, optimizer, criterion, train_dataloader,alpha,temp,l=3,beta=0.01, lamda=0.01,
                             show_interval=10, if_MMD = True, weight_Type = 'Avg', if_pearson = True,is_L1 = True,
                batch_size=Config.batch_size, device=None, reshape_flag=False):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    for sig_eeg, sig_delta, sig_theta, sig_alpha, sig_beta, \
        sig_gamma, sig_upper, target, sample_weight, class_weight in train_dataloader:
        sig_eeg = sig_eeg.to(device) # Batchsize * channel * 1 * points(32*20*1*512)
        sig_delta = sig_delta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
        sig_theta = sig_theta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
        sig_alpha = sig_alpha.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
        sig_beta = sig_beta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
        sig_gamma = sig_gamma.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
        sig_upper = sig_upper.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)

        if reshape_flag is True:
            sig_eeg = old_reshape_input(sig_eeg)#Batchsize * 1 *Channels * timepoin(128*20*1*1024)
            sig_delta = old_reshape_input(sig_delta)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
            sig_theta = old_reshape_input(sig_theta)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
            sig_alpha = old_reshape_input(sig_alpha)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
            sig_beta = old_reshape_input(sig_beta)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
            sig_gamma = old_reshape_input(sig_gamma)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
            sig_upper = old_reshape_input(sig_upper)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)

        target = target.to(device)#32,4
        target2 = torch.argmax(target, dim=1)#32,4

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward

        pred_x, pred_delta, pred_theta, pred_alpha, pred_beta, pred_gamma, pred_upper, \
        feature_map_x, feature_map_delta, feature_map_theta, feature_map_alpha, feature_map_beta, feature_map_gamma, feature_map_upper, \
        list_feature_map_x, list_feature_map_delta, list_feature_map_theta, list_feature_map_alpha, \
        list_feature_map_beta, list_feature_map_gamma, list_feature_map_upper, L1, trained_weight = model(sig_eeg, sig_delta,
                                                                                              sig_theta, sig_alpha, sig_beta,
                                                                                              sig_gamma, sig_upper)  # (64,4)


        CE_loss_x = criterion(pred_x, target2)

        CE_loss_delta = criterion(pred_delta, target2)
        CE_loss_theta = criterion(pred_theta, target2)
        CE_loss_alpha = criterion(pred_alpha, target2)
        CE_loss_beta = criterion(pred_beta, target2)
        CE_loss_gamma = criterion(pred_gamma, target2)
        CE_loss_upper = criterion(pred_upper, target2)

        loss_t = torch.stack([CE_loss_delta, CE_loss_theta, CE_loss_alpha, CE_loss_beta, CE_loss_gamma, CE_loss_upper], dim=0)
        loss_ts = []
        num_teacher = 6

        if weight_Type == 'CE':
            weight_teacher = (1.0 - F.softmax(loss_t, dim=0)) * 6

        elif weight_Type == 'Train':
            weight_teacher = F.softmax(trained_weight) * 6

        else:
            weight_teacher = torch.tensor([1, 1, 1, 1, 1, 1]).to(device)

        loss_ts.append((loss_KL( pred_delta, pred_x, T=temp)+(loss_KL( pred_x, pred_delta, T=temp)))/2)
        loss_ts.append((loss_KL( pred_theta, pred_x, T=temp)+(loss_KL( pred_x, pred_theta, T=temp)))/2)
        loss_ts.append((loss_KL( pred_alpha, pred_x, T=temp)+(loss_KL( pred_x, pred_alpha, T=temp)))/2)
        loss_ts.append((loss_KL( pred_beta, pred_x, T=temp)+(loss_KL( pred_x, pred_beta, T=temp)))/2)
        loss_ts.append((loss_KL( pred_gamma, pred_x, T=temp)+(loss_KL( pred_x, pred_gamma, T=temp)))/2)
        loss_ts.append((loss_KL( pred_upper, pred_x, T=temp)+(loss_KL( pred_x, pred_upper, T=temp)))/2)
        # loss_ts.append((loss_KL( pred_x, pred_delta, T=temp)))
        # loss_ts.append((loss_KL( pred_x, pred_theta, T=temp)))
        # loss_ts.append((loss_KL( pred_x, pred_alpha, T=temp)))
        # loss_ts.append((loss_KL( pred_x, pred_beta, T=temp)))
        # loss_ts.append((loss_KL( pred_x, pred_gamma, T=temp)))
        # loss_ts.append((loss_KL( pred_x, pred_upper, T=temp)))
        loss_st = torch.stack(loss_ts, dim=0)
        loss = torch.mul(weight_teacher, loss_st).sum()  / 6 * lamda

        if if_MMD:
            myMMD_loss = MMD_loss()
            for i in range(l):
                loss_mmd = [myMMD_loss(list_feature_map_delta[len(list_feature_map_delta) - 1 - i],
                                       list_feature_map_x[len(list_feature_map_delta) - 1 - i]),
                            myMMD_loss(list_feature_map_x[len(list_feature_map_delta) - 1 - i],
                                       list_feature_map_theta[len(list_feature_map_delta) - 1 - i]),
                            myMMD_loss(list_feature_map_alpha[len(list_feature_map_delta) - 1 - i],
                                       list_feature_map_x[len(list_feature_map_delta) - 1 - i]),
                            myMMD_loss(list_feature_map_x[len(list_feature_map_delta) - 1 - i],
                                       list_feature_map_beta[len(list_feature_map_delta) - 1 - i]),
                            myMMD_loss(list_feature_map_gamma[len(list_feature_map_delta) - 1 - i],
                                       list_feature_map_x[len(list_feature_map_delta) - 1 - i]),
                            myMMD_loss(list_feature_map_x[len(list_feature_map_delta) - 1 - i],
                                       list_feature_map_upper[len(list_feature_map_delta) - 1 - i])]
                loss_mmd = torch.stack(loss_mmd, dim=0)
                myloss = torch.mul(weight_teacher, loss_mmd).sum() / 6
                loss += myloss

        if if_pearson:
            for i in range(l):
                loss_pearsonr = [
                    torch.sum(torch.abs(pearsonr(list_feature_map_delta[len(list_feature_map_delta) - 1 - i],
                                                 list_feature_map_x[len(list_feature_map_delta) - 1 - i]))),

                    torch.sum(torch.abs(pearsonr(list_feature_map_x[len(list_feature_map_delta) - 1 - i],
                                                 list_feature_map_theta[len(list_feature_map_delta) - 1 - i]))),

                    torch.sum(torch.abs(pearsonr(list_feature_map_alpha[len(list_feature_map_delta) - 1 - i],
                                                 list_feature_map_x[len(list_feature_map_delta) - 1 - i]))),

                    torch.sum(torch.abs(pearsonr(list_feature_map_x[len(list_feature_map_delta) - 1 - i],
                                                 list_feature_map_beta[len(list_feature_map_delta) - 1 - i]))),

                    torch.sum(torch.abs(pearsonr(list_feature_map_gamma[len(list_feature_map_delta) - 1 - i],
                                                 list_feature_map_x[len(list_feature_map_delta) - 1 - i]))),

                    torch.sum(torch.abs(pearsonr(list_feature_map_x[len(list_feature_map_delta) - 1 - i],
                                                 list_feature_map_upper[len(list_feature_map_delta) - 1 - i])))]
                loss_pearsonr = torch.stack(loss_pearsonr, dim=0)
                myloss = torch.mul(weight_teacher, loss_pearsonr).sum() / 6
                loss += myloss

        loss += alpha * CE_loss_x
        if is_L1:
            # print(L1)

            loss += L1 * beta
        # print('L1:', L1 * beta)

        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        output = torch.argmax(pred_x, dim=1)

        precision, recall, f1, support = sm.precision_recall_fscore_support(target2.detach().cpu(), output.detach().cpu(), average='weighted')
        f1_meter += f1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e f1:%.3f precision:%.3f" % (it_count, loss.item(), f1, precision))
    return loss_meter / it_count, f1_meter / it_count


def train_epoch_enhanceTrain_tusz(model, optimizer, criterion, train_dataloader, alpha,temp,l=3,beta=0.01,lamda=0.01,
                                  show_interval=10, if_MMD = True, weight_Type = 'Avg', if_pearson = False, is_L1 = True,
                batch_size=Config.batch_size, device=None, reshape_flag=False):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    for sig_eeg, sig_delta, sig_theta, sig_alpha,  \
        sig_beta, sig_gamma, sig_upper, hotlabel, sample_weight, class_weight in train_dataloader:


        sig_eeg = sig_eeg.to(device) # Batchsize * channel * 1 * points(32*20*1*512)
        sig_delta = sig_delta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
        sig_theta = sig_theta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
        sig_alpha = sig_alpha.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
        sig_beta = sig_beta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
        sig_gamma = sig_gamma.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
        sig_upper = sig_upper.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)

        if reshape_flag is True:
            sig_eeg = old_reshape_input(sig_eeg)#Batchsize * 1 *Channels * timepoin(128*20*1*1024)
            sig_delta = old_reshape_input(sig_delta)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
            sig_theta = old_reshape_input(sig_theta)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
            sig_alpha = old_reshape_input(sig_alpha)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
            sig_beta = old_reshape_input(sig_beta)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
            sig_gamma = old_reshape_input(sig_gamma)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
            sig_upper = old_reshape_input(sig_upper)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)


        target = hotlabel.to(device)
        target2 = torch.argmax(target, dim=1)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward

        pred_x, pred_delta, pred_theta, pred_alpha, pred_beta, pred_gamma, pred_upper, \
        feature_map_x, feature_map_delta, feature_map_theta, feature_map_alpha, feature_map_beta, feature_map_gamma, feature_map_upper, \
        list_feature_map_x, list_feature_map_delta, list_feature_map_theta, list_feature_map_alpha, \
        list_feature_map_beta, list_feature_map_gamma, list_feature_map_upper, L1, trained_weight= model(sig_eeg, sig_delta,
                                                                                              sig_theta, sig_alpha, sig_beta,
                                                                                              sig_gamma, sig_upper)  # (64,4)

        CE_loss_x = criterion(pred_x, target2)
        CE_loss_delta = criterion(pred_delta, target2)
        CE_loss_theta = criterion(pred_theta, target2)
        CE_loss_alpha = criterion(pred_alpha, target2)
        CE_loss_beta = criterion(pred_beta, target2)
        CE_loss_gamma = criterion(pred_gamma, target2)
        CE_loss_upper = criterion(pred_upper, target2)

        loss_t = torch.stack([CE_loss_delta, CE_loss_theta, CE_loss_alpha, CE_loss_beta, CE_loss_gamma, CE_loss_upper], dim=0)
        num_teacher = 6
        loss_ts = []

        if weight_Type == 'CE':
            weight_teacher = (1.0 - F.softmax(loss_t, dim=0)) * 6

        elif weight_Type == 'Train':
            weight_teacher = F.softmax(trained_weight) * 6

        else:
            weight_teacher = torch.tensor([1, 1, 1, 1, 1, 1]).to(device)

        loss_ts.append((loss_KL( pred_delta, pred_x, T=temp)+(loss_KL( pred_x, pred_delta, T=temp)))/2)
        loss_ts.append((loss_KL( pred_theta, pred_x, T=temp)+(loss_KL( pred_x, pred_theta, T=temp)))/2)
        loss_ts.append((loss_KL( pred_alpha, pred_x, T=temp)+(loss_KL( pred_x, pred_alpha, T=temp)))/2)
        loss_ts.append((loss_KL( pred_beta, pred_x, T=temp)+(loss_KL( pred_x, pred_beta, T=temp)))/2)
        loss_ts.append((loss_KL( pred_gamma, pred_x, T=temp)+(loss_KL( pred_x, pred_gamma, T=temp)))/2)
        loss_ts.append((loss_KL( pred_upper, pred_x, T=temp)+(loss_KL( pred_x, pred_upper, T=temp)))/2)
        # loss_ts.append((loss_KL( pred_x, pred_delta, T=temp)))
        # loss_ts.append((loss_KL( pred_x, pred_theta, T=temp)))
        # loss_ts.append((loss_KL( pred_x, pred_alpha, T=temp)))
        # loss_ts.append((loss_KL( pred_x, pred_beta, T=temp)))
        # loss_ts.append((loss_KL( pred_x, pred_gamma, T=temp)))
        # loss_ts.append((loss_KL( pred_x, pred_upper, T=temp)))
        loss_st = torch.stack(loss_ts, dim=0)
        loss = torch.mul(weight_teacher, loss_st).sum()  / 6 *lamda

        if if_MMD:
            myMMD_loss = MMD_loss()
            for i in range(l):
                loss_mmd = [myMMD_loss(list_feature_map_delta[len(list_feature_map_delta) - 1 - i],
                                       list_feature_map_x[len(list_feature_map_delta) - 1 - i]),
                            myMMD_loss(list_feature_map_x[len(list_feature_map_delta) - 1 - i],
                                       list_feature_map_theta[len(list_feature_map_delta) - 1 - i]),
                            myMMD_loss(list_feature_map_alpha[len(list_feature_map_delta) - 1 - i],
                                       list_feature_map_x[len(list_feature_map_delta) - 1 - i]),
                            myMMD_loss(list_feature_map_x[len(list_feature_map_delta) - 1 - i],
                                       list_feature_map_beta[len(list_feature_map_delta) - 1 - i]),
                            myMMD_loss(list_feature_map_gamma[len(list_feature_map_delta) - 1 - i],
                                       list_feature_map_x[len(list_feature_map_delta) - 1 - i]),
                            myMMD_loss(list_feature_map_x[len(list_feature_map_delta) - 1 - i],
                                       list_feature_map_upper[len(list_feature_map_delta) - 1 - i])]
                loss_mmd = torch.stack(loss_mmd, dim=0)
                myloss = torch.mul(weight_teacher, loss_mmd).sum() / 6
                loss += myloss

        if if_pearson:

            for i in range(l):
                loss_pearsonr = [torch.sum(torch.abs(pearsonr(list_feature_map_delta[len(list_feature_map_delta) - 1 - i],
                                                              list_feature_map_x[len(list_feature_map_delta) - 1 - i]))),

                                 torch.sum(torch.abs(pearsonr(list_feature_map_x[len(list_feature_map_delta) - 1 - i],
                                                              list_feature_map_theta[len(list_feature_map_delta) - 1 - i]))),

                                 torch.sum(torch.abs(pearsonr(list_feature_map_alpha[len(list_feature_map_delta) - 1 - i],
                                                              list_feature_map_x[len(list_feature_map_delta) - 1 - i]))),

                                 torch.sum(torch.abs(pearsonr(list_feature_map_x[len(list_feature_map_delta) - 1 - i],
                                                               list_feature_map_beta[len(list_feature_map_delta) - 1 - i]))),

                                 torch.sum(torch.abs(pearsonr(list_feature_map_gamma[len(list_feature_map_delta) - 1 - i],
                                                              list_feature_map_x[len(list_feature_map_delta) - 1 - i]))),

                                 torch.sum(torch.abs(pearsonr(list_feature_map_x[len(list_feature_map_delta) - 1 - i],
                                                               list_feature_map_upper[len(list_feature_map_delta) - 1 - i])))]
                loss_pearsonr = torch.stack(loss_pearsonr, dim=0)
                myloss = torch.mul(weight_teacher, loss_pearsonr).sum() / 6
                loss += myloss

        loss += alpha * CE_loss_x
        if is_L1:
            loss += L1* beta
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        output = torch.argmax(pred_x, dim=1)

        precision, recall, f1, support = sm.precision_recall_fscore_support(target2.detach().cpu(), output.detach().cpu(), average='weighted')
        f1_meter += f1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e f1:%.3f precision:%.3f" % (it_count, loss.item(), f1, precision))
        # it_count = it_count * batch_size
    return loss_meter / it_count, f1_meter / it_count



def val_chsz(model, criterion, val_dataloader, device, reshape_flag=False):
    with torch.no_grad():
        model.eval()
        f1_meter, loss_meter, it_count, bca_meter = 0, 0, 0, 0
        output_meter = []
        target_meter=[]
        for sig_eeg, sig_delta, sig_theta, sig_alpha,  \
        sig_beta, sig_gamma, sig_upper, hotlabel, sample_weight, class_weight in val_dataloader:

            it_count += 1
            sig_eeg = sig_eeg.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            sig_delta = sig_delta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            sig_theta = sig_theta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            sig_alpha = sig_alpha.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            sig_beta = sig_beta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            sig_gamma = sig_gamma.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            sig_upper = sig_upper.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)

            if reshape_flag is True:
                sig_eeg = old_reshape_input(sig_eeg)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
                sig_delta = old_reshape_input(sig_delta)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
                sig_theta = old_reshape_input(sig_theta)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
                sig_alpha = old_reshape_input(sig_alpha)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
                sig_beta = old_reshape_input(sig_beta)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
                sig_gamma = old_reshape_input(sig_gamma)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
                sig_upper = old_reshape_input(sig_upper)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
            target = hotlabel.to(device)
            target2 = torch.argmax(target, dim=1)

            # _x, _featuremap, pred_label = model(inputs)
            pred_x, pred_delta, pred_theta, pred_alpha, pred_beta, pred_gamma, pred_upper, \
            feature_map_x, feature_map_delta, feature_map_theta, feature_map_alpha, feature_map_beta, feature_map_gamma, feature_map_upper, \
            list_feature_map_x, list_feature_map_delta, list_feature_map_theta, list_feature_map_alpha, \
            list_feature_map_beta, list_feature_map_gamma, list_feature_map_upper, L1, weight = model(sig_eeg, sig_delta,
                                                                                          sig_theta, sig_alpha,
                                                                                          sig_beta,
                                                                                          sig_gamma,
                                                                                          sig_upper)  # (64,4)
            target_meter.append(target2.cpu().numpy().tolist())

            output = torch.argmax(pred_x, dim=1)  # (1,)
            output = output.cpu()
            output_meter.append(output.cpu().numpy().tolist())
            my_loss = criterion(pred_x, target2)
            loss_meter += my_loss

            if it_count != 0 and it_count % 100 == 0:
                print("Validation: %d,loss:%.3e " % (it_count, my_loss.item()))

        val_loss = loss_meter / len(val_dataloader)

        target_meter = [item for sublist in target_meter for item in sublist]

        output_meter = [item for sublist in output_meter for item in sublist]

        val_f1 = sm.f1_score(target_meter, output_meter, average='weighted')
        val_bca = sm.balanced_accuracy_score(target_meter, output_meter)

    return output_meter, val_loss, val_f1, val_bca


def val_tusz(model, criterion, val_dataloader, device, reshape_flag=False):
    with torch.no_grad():
        model.eval()
        f1_meter, loss_meter, it_count, bca_meter = 0, 0, 0, 0
        output_meter = []
        target_meter = []
        for sig_eeg, sig_delta, sig_theta, sig_alpha,  \
        sig_beta, sig_gamma, sig_upper, hotlabel, sample_weight, class_weight in val_dataloader:

            it_count += 1
            sig_eeg = sig_eeg.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            sig_delta = sig_delta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            sig_theta = sig_theta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            sig_alpha = sig_alpha.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            sig_beta = sig_beta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            sig_gamma = sig_gamma.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            sig_upper = sig_upper.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)

            if reshape_flag is True:
                sig_eeg = old_reshape_input(sig_eeg)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
                sig_delta = old_reshape_input(sig_delta)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
                sig_theta = old_reshape_input(sig_theta)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
                sig_alpha = old_reshape_input(sig_alpha)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
                sig_beta = old_reshape_input(sig_beta)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
                sig_gamma = old_reshape_input(sig_gamma)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
                sig_upper = old_reshape_input(sig_upper)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)
            target = hotlabel.to(device)
            target2 = torch.argmax(target, dim=1)

            pred_x, pred_delta, pred_theta, pred_alpha, pred_beta, pred_gamma, pred_upper, \
            feature_map_x, feature_map_delta, feature_map_theta, feature_map_alpha, feature_map_beta, feature_map_gamma, feature_map_upper, \
            list_feature_map_x, list_feature_map_delta, list_feature_map_theta, list_feature_map_alpha, \
            list_feature_map_beta, list_feature_map_gamma, list_feature_map_upper, L1, weight = model(sig_eeg, sig_delta,
                                                                                          sig_theta, sig_alpha,
                                                                                          sig_beta,
                                                                                          sig_gamma,
                                                                                          sig_upper)  # (64,4)
            target_meter.append(target2.cpu().numpy().tolist())

            output = torch.argmax(pred_x, dim=1)  # (1,)
            output = output.cpu()
            output_meter.append(output.cpu().numpy().tolist())
            my_loss = criterion(pred_x, target2)
            loss_meter += my_loss

            if it_count != 0 and it_count % 100 == 0:
                print("Validation: %d,loss:%.3e " % (it_count, my_loss.item()))

        val_loss = loss_meter / len(val_dataloader)

        target_meter = [item for sublist in target_meter for item in sublist]

        output_meter = [item for sublist in output_meter for item in sublist]

        val_f1 = sm.f1_score(target_meter, output_meter, average='weighted')
        val_bca = sm.balanced_accuracy_score(target_meter, output_meter)
        print("validation:")
        print("BCA:", val_bca)
        print("f1", val_f1)

    return output_meter, val_loss, val_f1, val_bca
