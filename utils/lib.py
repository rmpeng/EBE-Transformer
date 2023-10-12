import time
import torch, shutil
import numpy as np
from my_config import Config
import os
import sklearn.metrics as sm

import os.path
from collections import Counter


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()


# define draw
def plotCurve(savename, x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(savename)
    plt.plot(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.plot(x2_vals, y2_vals, linestyle=':')
    plt.savefig(savename + '.png')
    plt.show()


# time consume
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


# save current weight and update new best weight
def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, Config.current_w)
    best_w = os.path.join(model_save_dir, Config.best_w)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)


def reshape_input(input):
    input = input.permute(0, 3, 2, 1)
    if input.shape[3] < 1024:
        valid = 1024 - input.shape[3]
        pat = input[:,:,:,:valid]
        input = torch.cat((input,pat),dim=3)
    return input

def reshape_input_new(input):
    input = input.permute(0, 2, 3, 1)
    if input.shape[3] < 1024:
        valid = 1024 - input.shape[3]
        pat = input[:,:,:,:valid]
        input = torch.cat((input,pat),dim=3)
    return input

def old_reshape_input(input):
    input = input.permute(0, 2, 1, 3)
    return input


def train_epoch(model, optimizer, criterion, train_dataloader, show_interval=10,
                batch_size=Config.batch_size, device=None, reshape_flag=True):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    for inputs, target, sweight, class_weight in train_dataloader:

        inputs = inputs.to(device) # Batchsize * channel * 1 * points(128*1024*1*20)
        if reshape_flag is True:
            inputs = old_reshape_input(inputs)#Batchsize * 1 *Channels * timepoin(128*20*1*1024)
        target = target.to(device)
        target2 = torch.argmax(target, dim=1)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        pred_label = model(inputs)  # (64,4)

        loss = criterion(pred_label, target2)

        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        output = torch.argmax(pred_label, dim=1)

        precision, recall, f1, support = sm.precision_recall_fscore_support(target2.detach().cpu(), output.detach().cpu(), average='weighted')
        f1_meter += f1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e f1:%.3f precision:%.3f" % (it_count, loss.item(), f1, precision))
        # it_count = it_count * batch_size
    return loss_meter / it_count, f1_meter / it_count

        # # zero the parameter gradients
        # # forward
        # temp_combine_rep, spec_combine_rep = model(inputs)  # (64,4)
        # return temp_combine_rep, spec_combine_rep

def train_epoch_mv_chsz(model, optimizer, criterion, train_dataloader, show_interval=10, input_type = 'sig',
                batch_size=Config.batch_size, device=None, reshape_flag=False, modal_type='EEG'):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    for seeg, semg, secg, feeg, femg, fecg, target, smapleweight, class_weight in train_dataloader:
    # for inputs, target, sweight, class_weight in train_dataloader:
        if input_type == 'sig':
            if modal_type == 'EEG':
                inputs = seeg
            elif modal_type == 'EMG':
                inputs = semg
            elif modal_type == 'ECG':
                inputs = secg
            else:
                inputs = torch.cat((seeg, semg,secg), 1)
        else:
            if modal_type == 'EEG':
                inputs = feeg

            elif modal_type == 'EMG':
                inputs = femg
            elif modal_type == 'ECG':
                inputs = fecg
            else:
                inputs = torch.cat((feeg, femg,fecg), 1)
            # inputs = torch.unsqueeze(inputs, dim=1)
            # inputs = torch.unsqueeze(inputs, dim=2)

        inputs = inputs.to(device) # Batchsize * channel * 1 * points(32*20*1*512)
        if reshape_flag is True:#EEGnet的时候要变
            inputs = old_reshape_input(inputs)#Batchsize * 1 *Channels * timepoin(128*20*1*1024)
        target = target.to(device)
        target2 = torch.argmax(target, dim=1)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        _x, _featuremap, pred_label = model(inputs)  # (64,4)

        loss = criterion(pred_label, target2)

        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        output = torch.argmax(pred_label, dim=1)

        precision, recall, f1, support = sm.precision_recall_fscore_support(target2.detach().cpu(), output.detach().cpu(), average='weighted')
        f1_meter += f1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e f1:%.3f precision:%.3f" % (it_count, loss.item(), f1, precision))
        # it_count = it_count * batch_size
    return loss_meter / it_count, f1_meter / it_count

        # # zero the parameter gradients
        # # forward
        # temp_combine_rep, spec_combine_rep = model(inputs)  # (64,4)
        # return temp_combine_rep, spec_combine_rep

def train_epoch_mv_tusz(model, optimizer, criterion, train_dataloader, show_interval=10, input_type = 'sig',
                batch_size=Config.batch_size, device=None, reshape_flag=False):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    for seeg, feeg,  target,sampleweight, class_weight in train_dataloader:
    # for inputs, target, sweight, class_weight in train_dataloader:
        if input_type == 'sig':
            inputs = seeg
        else:
            inputs = feeg
            # inputs = torch.unsqueeze(inputs, dim=1)
            # inputs = torch.unsqueeze(inputs, dim=2)

        inputs = inputs.to(device) # Batchsize * channel * 1 * points(128*1024*1*20)
        if reshape_flag is True:
            inputs = old_reshape_input(inputs)#Batchsize * 1 *Channels * timepoin(128*20*1*1024)
        target = target.to(device)
        target2 = torch.argmax(target, dim=1)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        _x, _featuremap,pred_label = model(inputs)  # (64,4)

        loss = criterion(pred_label, target2)

        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        output = torch.argmax(pred_label, dim=1)

        precision, recall, f1, support = sm.precision_recall_fscore_support(target2.detach().cpu(), output.detach().cpu(), average='weighted')
        f1_meter += f1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e f1:%.3f precision:%.3f" % (it_count, loss.item(), f1, precision))
        # it_count = it_count * batch_size
    return loss_meter / it_count, f1_meter / it_count

        # # zero the parameter gradients
        # # forward
        # temp_combine_rep, spec_combine_rep = model(inputs)  # (64,4)
        # return temp_combine_rep, spec_combine_rep


def val_former(model, criterion, val_data, device, reshape_flag=False):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(val_data.data, dtype=torch.float32)
        if reshape_flag is True:
            inputs = reshape_input(inputs)
        target = torch.tensor(val_data.org_label).long()  # (64,)

        inputs = inputs.to(device)
        pred_label = model(inputs)  # (64,4)
        output = torch.argmax(pred_label, dim=1)  # (64,)
        output = output.cpu()
        loss = criterion(pred_label.cpu(), target)
        bca = sm.balanced_accuracy_score(target, output)
        f1 = sm.f1_score(target, output, average='micro')

        print("validation:")
        print("BCA:", bca)
        print("f1", f1)

    return output, loss, f1, bca

def val(model, criterion, val_dataloader, device, reshape_flag=False):

    with torch.no_grad():
        model.eval()
        f1_meter, loss_meter, it_count, bca_meter = 0, 0, 0,0
        output_meter = []
        target_meter=[]
        for inputs, target, sweight, class_weight in val_dataloader:
            it_count += 1
            inputs = inputs.to(device)

            if reshape_flag is True:
                inputs = old_reshape_input(inputs)
            target = target.to(device)
            target2 = torch.argmax(target, dim=1)

            pred_label = model(inputs)
            target_meter.append(target2.cpu().numpy().tolist())

            output = torch.argmax(pred_label, dim=1)  # (1,)
            output = output.cpu()
            output_meter.append(output.cpu().numpy().tolist())
            my_loss = criterion(pred_label, target2)
            loss_meter += my_loss

            if it_count != 0 and it_count % 10 == 0:
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

def val_mv_chsz(model, criterion, val_dataloader, device, reshape_flag=False, modal_type='EEG' , input_type = 'sig'):
    with torch.no_grad():
        model.eval()
        f1_meter, loss_meter, it_count, bca_meter = 0, 0, 0, 0
        output_meter = []
        target_meter=[]
        for seeg, semg, secg, feeg, femg, fecg, target, sampleweight, class_weight in val_dataloader:
        # for inputs, target, sweight, class_weight in val_dataloader:
            if input_type == 'sig':
                if modal_type == 'EEG':
                    inputs = seeg
                elif modal_type == 'EMG':
                    inputs = semg
                elif modal_type == 'ECG':
                    inputs = secg
                else:
                    inputs = torch.cat((seeg, semg,secg), 1)
            else:
                if modal_type == 'EEG':
                    inputs = feeg
                elif modal_type == 'EMG':
                    inputs = femg
                elif modal_type == 'ECG':
                    inputs = fecg
                else:
                    inputs = torch.cat((feeg, femg,fecg), 1)
                # inputs = torch.unsqueeze(inputs, dim=1)
                # inputs = torch.unsqueeze(inputs, dim=2)

            it_count += 1
            inputs = inputs.to(device)

            if reshape_flag is True:
                inputs = old_reshape_input(inputs)
            target = target.to(device)
            target2 = torch.argmax(target, dim=1)

            _x, _featuremap, pred_label = model(inputs)
            target_meter.append(target2.cpu().numpy().tolist())

            output = torch.argmax(pred_label, dim=1)  # (1,)
            output = output.cpu()
            output_meter.append(output.cpu().numpy().tolist())
            my_loss = criterion(pred_label, target2)
            loss_meter += my_loss

            if it_count != 0 and it_count % 100 == 0:
                print("Validation: %d,loss:%.3e " % (it_count, my_loss.item()))

        val_loss = loss_meter / len(val_dataloader)

        target_meter = [item for sublist in target_meter for item in sublist]

        output_meter = [item for sublist in output_meter for item in sublist]

        val_f1 = sm.f1_score(target_meter, output_meter, average='weighted')
        val_bca = sm.balanced_accuracy_score(target_meter, output_meter)
        # print("validation:")
        # print("BCA:", val_bca)
        # print("f1", val_f1)

    return output_meter, val_loss, val_f1, val_bca


def val_mv_tusz(model, criterion, val_dataloader, device, reshape_flag=False, input_type = 'sig'):
    with torch.no_grad():
        model.eval()
        f1_meter, loss_meter, it_count, bca_meter = 0, 0, 0, 0
        output_meter = []
        target_meter = []
        for seeg, feeg,  target,sampleweight, class_weight in val_dataloader:
            # for inputs, target, sweight, class_weight in val_dataloader:
            if input_type == 'sig':
                inputs = seeg
            else:
                inputs = feeg
                # inputs = torch.unsqueeze(inputs, dim=1)
                # inputs = torch.unsqueeze(inputs, dim=2)

            it_count += 1
            inputs = inputs.to(device)

            if reshape_flag is True:
                inputs = old_reshape_input(inputs)
            target = target.to(device)
            target2 = torch.argmax(target, dim=1)

            _x, _featuremap, pred_label = model(inputs)
            target_meter.append(target2.cpu().numpy().tolist())

            output = torch.argmax(pred_label, dim=1)  # (1,)
            output = output.cpu()
            output_meter.append(output.cpu().numpy().tolist())
            my_loss = criterion(pred_label, target2)
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


def train_epoch_mv_chsz_new(model, optimizer, criterion, train_dataloader, show_interval=10, input_type = 'sig',
                batch_size=Config.batch_size, device=None, reshape_flag=False, modal_type='EEG'):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    # for seeg, semg, secg, feeg, femg, fecg, target, smapleweight, class_weight in train_dataloader:

    # for seeg, semg, secg, feeg, femg, fecg, target, smapleweight, class_weight in train_dataloader:
    for seeg, semg, secg, feeg, femg, fecg, target, sample_weight, class_weight,\
               seeg_neg, semg_neg, secg_neg, feeg_neg, femg_neg, fecg_neg, target_neg, \
               sample_weight_neg, class_weight_neg in train_dataloader:
        if input_type == 'sig':
            if modal_type == 'EEG':
                inputs = seeg
            elif modal_type == 'EMG':
                inputs = semg
            elif modal_type == 'ECG':
                inputs = secg
            else:
                inputs = torch.cat((seeg, semg,secg), 1)
        else:
            if modal_type == 'EEG':
                inputs = feeg

            elif modal_type == 'EMG':
                inputs = femg
            elif modal_type == 'ECG':
                inputs = fecg
            else:
                inputs = torch.cat((feeg, femg,fecg), 1)
            # inputs = torch.unsqueeze(inputs, dim=1)
            # inputs = torch.unsqueeze(inputs, dim=2)

        inputs = inputs.to(device) # Batchsize * channel * 1 * points(32*20*1*512)
        if reshape_flag is True:
            inputs = old_reshape_input(inputs)#Batchsize * 1 *Channels * timepoin(128*20*1*1024)
        target = target.to(device)
        target2 = torch.argmax(target, dim=1)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        _x, _featuremap, pred_label = model(inputs)  # (64,4)

        loss = criterion(pred_label, target2)

        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        output = torch.argmax(pred_label, dim=1)

        precision, recall, f1, support = sm.precision_recall_fscore_support(target2.detach().cpu(), output.detach().cpu(), average='weighted')
        f1_meter += f1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e f1:%.3f precision:%.3f" % (it_count, loss.item(), f1, precision))
        # it_count = it_count * batch_size
    return loss_meter / it_count, f1_meter / it_count

        # # zero the parameter gradients
        # # forward
        # temp_combine_rep, spec_combine_rep = model(inputs)  # (64,4)
        # return temp_combine_rep, spec_combine_rep

def train_epoch_mv_tusz_new(model, optimizer, criterion, train_dataloader, show_interval=10, input_type = 'sig',
                batch_size=Config.batch_size, device=None, reshape_flag=False):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    # for seeg, feeg,  target,sampleweight, class_weight in train_dataloader:
    for seeg, feeg,  target, sample_weight, class_weight, \
            seeg_neg, feeg_neg, target_neg, sample_weight_neg, class_weight_neg in train_dataloader:
        if input_type == 'sig':
            inputs = seeg
        else:
            inputs = feeg
            # inputs = torch.unsqueeze(inputs, dim=1)
            # inputs = torch.unsqueeze(inputs, dim=2)

        inputs = inputs.to(device) # Batchsize * channel * 1 * points(128*1024*1*20)
        if reshape_flag is True:
            inputs = old_reshape_input(inputs)#Batchsize * 1 *Channels * timepoin(128*20*1*1024)
        target = target.to(device)
        target2 = torch.argmax(target, dim=1)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        _x, _featuremap,pred_label = model(inputs)  # (64,4)

        loss = criterion(pred_label, target2)

        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        output = torch.argmax(pred_label, dim=1)

        precision, recall, f1, support = sm.precision_recall_fscore_support(target2.detach().cpu(), output.detach().cpu(), average='weighted')
        f1_meter += f1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e f1:%.3f precision:%.3f" % (it_count, loss.item(), f1, precision))
        # it_count = it_count * batch_size
    return loss_meter / it_count, f1_meter / it_count

        # # zero the parameter gradients
        # # forward
        # temp_combine_rep, spec_combine_rep = model(inputs)  # (64,4)
        # return temp_combine_rep, spec_combine_rep


def val_mv_chsz_new(model, criterion, val_dataloader, device, reshape_flag=False, modal_type='EEG' , input_type = 'sig'):
    with torch.no_grad():
        model.eval()
        f1_meter, loss_meter, it_count, bca_meter = 0, 0, 0, 0
        output_meter = []
        target_meter=[]
        for seeg, semg, secg, feeg, femg, fecg, target, sample_weight, class_weight,\
               seeg_neg, semg_neg, secg_neg, feeg_neg, femg_neg, fecg_neg, target_neg, \
               sample_weight_neg, class_weight_neg in val_dataloader:
        # for inputs, target, sweight, class_weight in val_dataloader:
            if input_type == 'sig':
                if modal_type == 'EEG':
                    inputs = seeg
                elif modal_type == 'EMG':
                    inputs = semg
                elif modal_type == 'ECG':
                    inputs = secg
                else:
                    inputs = torch.cat((seeg, semg,secg), 1)
            else:
                if modal_type == 'EEG':
                    inputs = feeg
                elif modal_type == 'EMG':
                    inputs = femg
                elif modal_type == 'ECG':
                    inputs = fecg
                else:
                    inputs = torch.cat((feeg, femg,fecg), 1)
                # inputs = torch.unsqueeze(inputs, dim=1)
                # inputs = torch.unsqueeze(inputs, dim=2)

            it_count += 1
            inputs = inputs.to(device)

            if reshape_flag is True:
                inputs = old_reshape_input(inputs)
            target = target.to(device)
            target2 = torch.argmax(target, dim=1)

            _x, _featuremap, pred_label = model(inputs)
            target_meter.append(target2.cpu().numpy().tolist())

            output = torch.argmax(pred_label, dim=1)  # (1,)
            output = output.cpu()
            output_meter.append(output.cpu().numpy().tolist())
            my_loss = criterion(pred_label, target2)
            loss_meter += my_loss

            if it_count != 0 and it_count % 10 == 0:
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


def val_mv_tusz_new(model, criterion, val_dataloader, device, reshape_flag=False, input_type = 'sig'):
    with torch.no_grad():
        model.eval()
        f1_meter, loss_meter, it_count, bca_meter = 0, 0, 0, 0
        output_meter = []
        target_meter = []
        for seeg, feeg,  target, sample_weight, class_weight, \
            seeg_neg, feeg_neg, target_neg, sample_weight_neg, class_weight_neg in val_dataloader:
            # for inputs, target, sweight, class_weight in val_dataloader:
            if input_type == 'sig':
                inputs = seeg
            else:
                inputs = feeg
                # inputs = torch.unsqueeze(inputs, dim=1)
                # inputs = torch.unsqueeze(inputs, dim=2)

            it_count += 1
            inputs = inputs.to(device)

            if reshape_flag is True:
                inputs = old_reshape_input(inputs)
            target = target.to(device)
            target2 = torch.argmax(target, dim=1)

            _x, _featuremap, pred_label = model(inputs)
            target_meter.append(target2.cpu().numpy().tolist())

            output = torch.argmax(pred_label, dim=1)  # (1,)
            output = output.cpu()
            output_meter.append(output.cpu().numpy().tolist())
            my_loss = criterion(pred_label, target2)
            loss_meter += my_loss

            if it_count != 0 and it_count % 10 == 0:
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




def vote(pred_y):  # 预测结果投票
    trails = 1
    new_y = torch.argmax(pred_y, dim=1)
    vote_label = Counter(new_y.numpy()).most_common(1)[0][0]
    vote_label_series = vote_label * torch.ones(trails)
    return vote_label_series

def vote_np(pred_y):  # 预测结果投票
    trails = 1
    new_y = np.argmax(pred_y, axis=1)
    vote_label = Counter(new_y).most_common(1)[0][0]
    vote_label_series = vote_label * np.ones(trails)
    return vote_label_series

def ensemble_test(model, alpha_list, test_data, device):
    pred_arr = 0
    for alpha in alpha_list:
        model.Enk_Layer.alpha=alpha
        subpred_label = model(test_data.to(torch.float32).detach().to(device)).detach().cpu().numpy()
        pred_arr += subpred_label

    pred_label_arr = pred_arr / len(alpha_list)
    pred_y = torch.from_numpy(pred_label_arr)

    return pred_y
