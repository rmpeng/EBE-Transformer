import torch
from torch.nn import functional as F
from my_config import Config
import sklearn.metrics as sm

def old_reshape_input(input):
    input = input.permute(0, 2, 1, 3)
    return input

def train_epoch_chsz(model, optimizer, criterion, train_dataloader,
                             show_interval=10, signaltype ='data',
                batch_size=Config.batch_size, device=None, reshape_flag=False):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    for sig_eeg, sig_delta, sig_theta, sig_alpha, sig_beta, \
        sig_gamma, sig_upper, target, sample_weight, class_weight in train_dataloader:

        if signaltype == 'data':
            sig_eeg = sig_eeg.to(device) # Batchsize * channel * 1 * points(32*20*1*512)
        elif  signaltype == 'delta':
            sig_eeg = sig_delta.to(device) # Batchsize * channel * 1 * points(32*20*1*512)
        elif  signaltype == 'alpha':
            sig_eeg = sig_alpha.to(device) # Batchsize * channel * 1 * points(32*20*1*512)
        elif  signaltype == 'beta':
            sig_eeg = sig_beta.to(device) # Batchsize * channel * 1 * points(32*20*1*512)
        elif  signaltype == 'theta':
            sig_eeg = sig_theta.to(device) # Batchsize * channel * 1 * points(32*20*1*512)
        elif signaltype == 'gamma':
            sig_eeg = sig_gamma.to(device) # Batchsize * channel * 1 * points(32*20*1*512)
        else:
            sig_eeg = sig_upper.to(device) # Batchsize * channel * 1 * points(32*20*1*512)


        if reshape_flag is True:
            sig_eeg = old_reshape_input(sig_eeg)#Batchsize * 1 *Channels * timepoin(128*20*1*1024)


        target = target.to(device)#32,4
        target2 = torch.argmax(target, dim=1)#32,4

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward

        x, feature_map_x, pred_x= model(sig_eeg)  # (64,4)


        loss = criterion(pred_x, target2)

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


def train_epoch_tusz(model, optimizer, criterion, train_dataloader,
                                  show_interval=10, signaltype ='data',
                batch_size=Config.batch_size, device=None, reshape_flag=False):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    for sig_eeg, sig_delta, sig_theta, sig_alpha,  \
        sig_beta, sig_gamma, sig_upper, hotlabel, sample_weight, class_weight in train_dataloader:

        if signaltype == 'data':
            sig_eeg = sig_eeg.to(device) # Batchsize * channel * 1 * points(32*20*1*512)
        elif  signaltype == 'delta':
            sig_eeg = sig_delta.to(device) # Batchsize * channel * 1 * points(32*20*1*512)
        elif  signaltype == 'alpha':
            sig_eeg = sig_alpha.to(device) # Batchsize * channel * 1 * points(32*20*1*512)
        elif  signaltype == 'beta':
            sig_eeg = sig_beta.to(device) # Batchsize * channel * 1 * points(32*20*1*512)
        elif  signaltype == 'theta':
            sig_eeg = sig_theta.to(device) # Batchsize * channel * 1 * points(32*20*1*512)
        elif signaltype == 'gamma':
            sig_eeg = sig_gamma.to(device) # Batchsize * channel * 1 * points(32*20*1*512)
        else:
            sig_eeg = sig_upper.to(device) # Batchsize * channel * 1 * points(32*20*1*512)


        if reshape_flag is True:
            sig_eeg = old_reshape_input(sig_eeg)#Batchsize * 1 *Channels * timepoin(128*20*1*1024)



        target = hotlabel.to(device)
        target2 = torch.argmax(target, dim=1)

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        x, feature_map_x, pred_x= model(sig_eeg)  # (64,4)

        loss = criterion(pred_x, target2)

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

        # # zero the parameter gradients
        # # forward
        # temp_combine_rep, spec_combine_rep = model(inputs)  # (64,4)
        # return temp_combine_rep, spec_combine_rep


def val_chsz(model, criterion, val_dataloader, device, signaltype ='data',reshape_flag=False):
    with torch.no_grad():
        model.eval()
        f1_meter, loss_meter, it_count, bca_meter = 0, 0, 0, 0
        output_meter = []
        target_meter=[]
        for sig_eeg, sig_delta, sig_theta, sig_alpha,  \
        sig_beta, sig_gamma, sig_upper, hotlabel, sample_weight, class_weight in val_dataloader:

            it_count += 1
            if signaltype == 'data':
                sig_eeg = sig_eeg.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            elif signaltype == 'delta':
                sig_eeg = sig_delta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            elif signaltype == 'alpha':
                sig_eeg = sig_alpha.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            elif signaltype == 'beta':
                sig_eeg = sig_beta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            elif signaltype == 'theta':
                sig_eeg = sig_theta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            elif signaltype == 'gamma':
                sig_eeg = sig_gamma.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            else:
                sig_eeg = sig_upper.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)


            if reshape_flag is True:
                sig_eeg = old_reshape_input(sig_eeg)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)

            target = hotlabel.to(device)
            target2 = torch.argmax(target, dim=1)

            # _x, _featuremap, pred_label = model(inputs)
            x, feature_map_x, pred_x = model(sig_eeg)  # (64,4)
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


def val_tusz(model, criterion, val_dataloader, device, signaltype ='data',reshape_flag=False):
    with torch.no_grad():
        model.eval()
        f1_meter, loss_meter, it_count, bca_meter = 0, 0, 0, 0
        output_meter = []
        target_meter = []
        for sig_eeg, sig_delta, sig_theta, sig_alpha,  \
        sig_beta, sig_gamma, sig_upper, hotlabel, sample_weight, class_weight in val_dataloader:

            it_count += 1
            if signaltype == 'data':
                sig_eeg = sig_eeg.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            elif signaltype == 'delta':
                sig_eeg = sig_delta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            elif signaltype == 'alpha':
                sig_eeg = sig_alpha.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            elif signaltype == 'beta':
                sig_eeg = sig_beta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            elif signaltype == 'theta':
                sig_eeg = sig_theta.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            elif signaltype == 'gamma':
                sig_eeg = sig_gamma.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)
            else:
                sig_eeg = sig_upper.to(device)  # Batchsize * channel * 1 * points(32*20*1*512)


            if reshape_flag is True:
                sig_eeg = old_reshape_input(sig_eeg)  # Batchsize * 1 *Channels * timepoin(128*20*1*1024)

            target = hotlabel.to(device)
            target2 = torch.argmax(target, dim=1)

            x, feature_map_x, pred_x = model(sig_eeg)  # (64,4)
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
