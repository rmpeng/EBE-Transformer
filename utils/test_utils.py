import torch
from collections import Counter
import numpy as np

def output_ONE(model, dataloader, device):
    event_pred = []
    event_label = []
    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1].to(device)
        if len(x.size())==0:
            continue
        # pred_y, pred_low, pred_high, feature_map_x, feature_map_low, \
        # feature_map_high = model(x.to(torch.float32).to(device))
        pred_1, pred_2, pred_3, pred_y = model(x.to(torch.float32).to(device))  # (64,4)

        # pred = vote(pred_y.detach().cpu())
        event_pred.append(pred_y.detach().cpu())
        event_label = y[0].cpu().numpy()
        # event_label.append(y.detach().cpu().numpy())
    event_pred = torch.cat(event_pred)
    # print(event_pred.size())
    event_pred = vote(event_pred.detach().cpu())
    event_pred = np.array(event_pred).flatten()
    event_label = np.array(event_label).flatten()
    # print(event_pred.shape,event_label.shape)
    # print(event_label)
    return event_pred, event_label


def output_BYOT(model, dataloader, device):
    event_pred = []
    event_label = []
    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1].to(device)
        if len(x.size())==0:
            continue
        # pred_y, pred_low, pred_high, feature_map_x, feature_map_low, \
        # feature_map_high = model(x.to(torch.float32).to(device))
        pred_y, feature_map_x, list_feature_map_x, list_pred_x= model(x.to(torch.float32).to(device))  # (64,4)

        # pred = vote(pred_y.detach().cpu())
        event_pred.append(pred_y.detach().cpu())
        event_label = y[0].cpu().numpy()
        # event_label.append(y.detach().cpu().numpy())
    event_pred = torch.cat(event_pred)
    # print(event_pred.size())
    event_pred = vote(event_pred.detach().cpu())
    event_pred = np.array(event_pred).flatten()
    event_label = np.array(event_label).flatten()
    # print(event_pred.shape,event_label.shape)
    # print(event_label)
    return event_pred, event_label

def output_org(model, dataloader, device):
    event_pred = []
    event_label = []
    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1].to(device)
        if len(x.size())==0:
            continue
        # pred_y, pred_low, pred_high, feature_map_x, feature_map_low, \
        # feature_map_high = model(x.to(torch.float32).to(device))
        x, feature_map_x, pred_y = model(x.to(torch.float32).to(device))  # (64,4)


        # pred = vote(pred_y.detach().cpu())
        event_pred.append(pred_y.detach().cpu())
        event_label = y[0].cpu().numpy()
        # event_label.append(y.detach().cpu().numpy())
    event_pred = torch.cat(event_pred)
    # print(event_pred.size())
    event_pred = vote(event_pred.detach().cpu())
    event_pred = np.array(event_pred).flatten()
    event_label = np.array(event_label).flatten()
    # print(event_pred.shape,event_label.shape)
    # print(event_label)
    return event_pred, event_label

def output(model, dataloader, device):
    event_pred = []
    event_label = []
    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1].to(device)
        if len(x.size())==0:
            continue
        # pred_y, pred_low, pred_high, feature_map_x, feature_map_low, feature_map_high = model(x.to(torch.float32).to(device))
        pred_y, pred_low, pred_high, feature_map_x, feature_map_low, feature_map_high, \
            list_feature_map_x, list_feature_map_low, list_feature_map_high = model(x.to(torch.float32).to(device), x.to(torch.float32).to(device),
                                                                                              x.to(torch.float32).to(device), x.to(torch.float32).to(device),
                                                                                              x.to(torch.float32).to(device), x.to(torch.float32).to(device),
                                                                                              x.to(torch.float32).to(device))  # (64,4)        event_pred.append(pred_y.detach().cpu())
        event_pred.append(pred_y.detach().cpu())
        event_label = y[0].cpu().numpy()
        # event_label.append(y.detach().cpu().numpy())
    event_pred = torch.cat(event_pred)
    # print(event_pred.size())
    event_pred = vote(event_pred.detach().cpu())
    event_pred = np.array(event_pred).flatten()
    event_label = np.array(event_label).flatten()
    # print(event_pred.shape,event_label.shape)
    # print(event_label)
    return event_pred, event_label

def output_reconstruct(model, dataloader, device):
    event_pred = []
    event_label = []
    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1].to(device)
        if len(x.size())==0:
            continue
        # pred_y, pred_low, pred_high, feature_map_x, feature_map_low, feature_map_high = model(x.to(torch.float32).to(device))
        pred_y, pred_reconstruct, feature_map_x, feature_map_low, feature_map_high, \
            list_feature_map_x, list_feature_map_low, list_feature_map_high = model(x.to(torch.float32).to(device), x.to(torch.float32).to(device),
                                                                                              x.to(torch.float32).to(device), x.to(torch.float32).to(device),
                                                                                              x.to(torch.float32).to(device), x.to(torch.float32).to(device),
                                                                                              x.to(torch.float32).to(device))  # (64,4)        event_pred.append(pred_y.detach().cpu())
        event_pred.append(pred_y.detach().cpu())
        event_label = y[0].cpu().numpy()
        # event_label.append(y.detach().cpu().numpy())
    event_pred = torch.cat(event_pred)
    # print(event_pred.size())
    event_pred = vote(event_pred.detach().cpu())
    event_pred = np.array(event_pred).flatten()
    event_label = np.array(event_label).flatten()
    # print(event_pred.shape,event_label.shape)
    # print(event_label)
    return event_pred, event_label


def vote(pred_y):  # 预测结果投票
    trails = 1
    new_y = torch.argmax(pred_y, dim=1)
    vote_label = Counter(new_y.numpy()).most_common(1)[0][0]
    vote_label_series = vote_label * torch.ones(trails)
    return vote_label_series


def chbmit_output(model, dataloader, device):
    pred_label = []
    target_label = []
    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1]
        if len(x.size())==0:
            continue
        pred_y, pred_low, pred_high, feature_map_x, feature_map_low, feature_map_high, \
            list_feature_map_x, list_feature_map_low, list_feature_map_high = model(x.to(torch.float32).to(device), x.to(torch.float32).to(device),
                                                                                              x.to(torch.float32).to(device), x.to(torch.float32).to(device),
                                                                                              x.to(torch.float32).to(device), x.to(torch.float32).to(device),
                                                                                              x.to(torch.float32).to(device))  # (64,4)        event_pred.append(pred_y.detach().cpu())
        new_y = torch.argmax(pred_y, dim=1)
        pred_label.append(new_y.detach().cpu())
        target_label.append(y)

    pred_label = np.concatenate(pred_label, axis=0)
    target_label = np.concatenate(target_label, axis=0)
    # print(event_pred.shape,event_label.shape)
    # print(event_label)
    return pred_label, target_label


def chbmit_output_org(model, dataloader, device):
    pred_label = []
    target_label = []
    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1]
        if len(x.size())==0:
            continue
        x, feature_map_x, pred_y = model(x.to(torch.float32).to(device))  # (64,4)

        new_y = torch.argmax(pred_y, dim=1)
        pred_label.append(new_y.detach().cpu())
        target_label.append(y)

    pred_label = np.concatenate(pred_label, axis=0)
    target_label = np.concatenate(target_label, axis=0)
    # print(event_pred.shape,event_label.shape)
    # print(event_label)
    return pred_label, target_label

def output_waves(model, dataloader, device):
    event_pred = []
    event_label = []
    for i, batch in enumerate(dataloader):
        x, sig_delta, sig_theta, \
        sig_alpha, sig_beta, sig_gamma, sig_upper, y = batch[0].to(device), batch[1].to(device), \
                                                               batch[2].to(device), batch[3].to(device),\
                                                               batch[4].to(device), batch[5].to(device),\
                                                               batch[6].to(device), batch[7].to(device)
        # x, y = batch[0].to(device), batch[1].to(device)
        if len(x.size())==0:
            continue
        # pred_y, pred_low, pred_high, feature_map_x, feature_map_low, \
        # feature_map_high = model(x.to(torch.float32).to(device))
        # x, feature_map_x, pred_y = model(x.to(torch.float32).to(device))  # (64,4)
        pred_y = model(sig_delta.to(torch.float32).to(device),
                       sig_theta.to(torch.float32).to(device), sig_alpha.to(torch.float32).to(device),
                       sig_beta.to(torch.float32).to(device), sig_gamma.to(torch.float32).to(device),
                       sig_upper.to(torch.float32).to(device))  # (64,4)


        # pred = vote(pred_y.detach().cpu())
        event_pred.append(pred_y.detach().cpu())
        event_label = y[0].cpu().numpy()
        # event_label.append(y.detach().cpu().numpy())
    event_pred = torch.cat(event_pred)
    # print(event_pred.size())
    event_pred = vote(event_pred.detach().cpu())
    event_pred = np.array(event_pred).flatten()
    event_label = np.array(event_label).flatten()
    # print(event_pred.shape,event_label.shape)
    # print(event_label)
    return event_pred, event_label

def output_waves_datt_analysis(model, dataloader, device,data_i):
    event_pred = []
    event_label = []
    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1].to(device)
        if len(x.size())==0:
            continue
        # pred_y, pred_low, pred_high, feature_map_x, feature_map_low, feature_map_high = model(x.to(torch.float32).to(device))
        pred_y, pred_delta, pred_theta, pred_alpha, pred_beta, pred_gamma, pred_upper, \
        feature_map_x, feature_map_delta, feature_map_theta, feature_map_alpha, feature_map_beta, feature_map_gamma, feature_map_upper, \
        list_feature_map_x, list_feature_map_delta, list_feature_map_theta, list_feature_map_alpha, \
        list_feature_map_beta, list_feature_map_gamma, list_feature_map_upper, L1, weight = model(x.to(torch.float32).to(device), x.to(torch.float32).to(device),
                                                                                              x.to(torch.float32).to(device), x.to(torch.float32).to(device),
                                                                                              x.to(torch.float32).to(device), x.to(torch.float32).to(device),
                                                                                              x.to(torch.float32).to(device))  # (64,4)        event_pred.append(pred_y.detach().cpu())
        pred_list = [pred_y, pred_delta, pred_theta, pred_alpha, pred_beta, pred_gamma, pred_upper]
        mypred_y = pred_list[data_i]
        # print(mypred_y.shape)
        event_pred.append(mypred_y.detach().cpu())
        event_label = y[0].cpu().numpy()
        # event_label.append(y.detach().cpu().numpy())
    event_pred = torch.cat(event_pred)
    # print(event_pred.size())
    event_pred = vote(event_pred.detach().cpu())
    event_pred = np.array(event_pred).flatten()
    event_label = np.array(event_label).flatten()
    # print(event_pred.shape,event_label.shape)
    # print(event_label)
    return event_pred, event_label


def output_waves_datt(model, dataloader, device):
    event_pred = []
    event_label = []
    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1].to(device)
        if len(x.size())==0:
            continue
        # pred_y, pred_low, pred_high, feature_map_x, feature_map_low, feature_map_high = model(x.to(torch.float32).to(device))
        pred_y, pred_delta, pred_theta, pred_alpha, pred_beta, pred_gamma, pred_upper, \
        feature_map_x, feature_map_delta, feature_map_theta, feature_map_alpha, feature_map_beta, feature_map_gamma, feature_map_upper, \
        list_feature_map_x, list_feature_map_delta, list_feature_map_theta, list_feature_map_alpha, \
        list_feature_map_beta, list_feature_map_gamma, list_feature_map_upper, L1, weight = model(x.to(torch.float32).to(device), x.to(torch.float32).to(device),
                                                                                              x.to(torch.float32).to(device), x.to(torch.float32).to(device),
                                                                                              x.to(torch.float32).to(device), x.to(torch.float32).to(device),
                                                                                              x.to(torch.float32).to(device))  # (64,4)        event_pred.append(pred_y.detach().cpu())
        event_pred.append(pred_y.detach().cpu())
        event_label = y[0].cpu().numpy()
        # event_label.append(y.detach().cpu().numpy())
    event_pred = torch.cat(event_pred)
    # print(event_pred.size())
    event_pred = vote(event_pred.detach().cpu())
    event_pred = np.array(event_pred).flatten()
    event_label = np.array(event_label).flatten()
    # print(event_pred.shape,event_label.shape)
    # print(event_label)
    return event_pred, event_label

def output_waves_datt_3waves(model, dataloader, device):
    event_pred = []
    event_label = []
    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1].to(device)
        if len(x.size())==0:
            continue
        # pred_y, pred_low, pred_high, feature_map_x, feature_map_low, feature_map_high = model(x.to(torch.float32).to(device))
        pred_y,  pred_theta, pred_alpha,  pred_upper, feature_map_x, feature_map_theta, feature_map_alpha, \
        feature_map_upper, list_feature_map_x, list_feature_map_theta, list_feature_map_alpha, \
         list_feature_map_upper, L1, weight = model(x.to(torch.float32).to(device),
                                                    x.to(torch.float32).to(device),
                                                    x.to(torch.float32).to(device),
                                                    x.to(torch.float32).to(device))  # (64,4)        event_pred.append(pred_y.detach().cpu())
        event_pred.append(pred_y.detach().cpu())
        event_label = y[0].cpu().numpy()
        # event_label.append(y.detach().cpu().numpy())
    event_pred = torch.cat(event_pred)
    # print(event_pred.size())
    event_pred = vote(event_pred.detach().cpu())
    event_pred = np.array(event_pred).flatten()
    event_label = np.array(event_label).flatten()
    # print(event_pred.shape,event_label.shape)
    # print(event_label)
    return event_pred, event_label

def output_waves_datt_2waves(model, dataloader, device):
    event_pred = []
    event_label = []
    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1].to(device)
        if len(x.size())==0:
            continue
        # pred_y, pred_low, pred_high, feature_map_x, feature_map_low, feature_map_high = model(x.to(torch.float32).to(device))
        pred_y, pred_alpha, pred_upper, feature_map_x, feature_map_alpha, \
        feature_map_upper, list_feature_map_x, list_feature_map_alpha, \
         list_feature_map_upper, L1, weight = model(x.to(torch.float32).to(device),
                                                    x.to(torch.float32).to(device),
                                                    x.to(torch.float32).to(device))  # (64,4)        event_pred.append(pred_y.detach().cpu())
        event_pred.append(pred_y.detach().cpu())
        event_label = y[0].cpu().numpy()
        # event_label.append(y.detach().cpu().numpy())
    event_pred = torch.cat(event_pred)
    # print(event_pred.size())
    event_pred = vote(event_pred.detach().cpu())
    event_pred = np.array(event_pred).flatten()
    event_label = np.array(event_label).flatten()
    # print(event_pred.shape,event_label.shape)
    # print(event_label)
    return event_pred, event_label


def BYOT_output(model, dataloader, device):
    event_pred = []
    event_label = []
    for i, batch in enumerate(dataloader):
        x, y = batch[0].to(device), batch[1]
        if len(x.size())==0:
            continue
        pred_y, feature_map_x, list_feature_map_x, list_pred_shallow_x = model(x.to(torch.float32).to(device))  # (64,4)        event_pred.append(pred_y.detach().cpu())
        event_pred.append(pred_y.detach().cpu())
        event_label = y[0].cpu().numpy()
        # event_label.append(y.detach().cpu().numpy())
    event_pred = torch.cat(event_pred)
    # print(event_pred.size())
    event_pred = vote(event_pred.detach().cpu())
    event_pred = np.array(event_pred).flatten()
    event_label = np.array(event_label).flatten()

    return event_pred, event_label

