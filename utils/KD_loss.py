import torch.nn.functional as F
import torch.nn as nn
#Kullback-Leibler divergence
def loss_KD(student_output, teacher_output, labels, alpha=0.5, T=3):
    alpha = alpha
    temperature = T
    y_target = F.softmax(teacher_output/temperature, dim=1)
    y_student = F.log_softmax(student_output/temperature, dim=1)
    L_soft = nn.KLDivLoss()(y_student,y_target)
    L_hard = F.cross_entropy(student_output,labels)
    KD_loss = L_soft * (alpha * temperature * temperature) + L_hard * (1.-alpha)
    return KD_loss

#Jessen-Shannon Divergence
def JSD_loss(student_output, teacher_output, labels, T=6):
    alpha = 0.9
    output1 = F.log_softmax(student_output / T, dim=1)
    output2 = F.log_softmax(teacher_output / T, dim=1)

    M = 0.5*(F.softmax(teacher_output/T, dim=1)+F.softmax(student_output/T, dim=1))

    JSD_loss = 0.5*(nn.KLDivLoss()(output1,M) +  nn.KLDivLoss()(output2,M)) * (alpha * T * T) + \
              0.5*(F.cross_entropy(output1, labels)+F.cross_entropy(output2, labels)) * (1. - alpha)
    return JSD_loss

# input = self.softmax(x)
def sharpen(x, T):
    temp = x**(1/T)
    return temp / temp.sum(axis=1, keepdims=True)

def sharpen_KD_loss(student_output, teacher_output, labels):
    alpha = 0.9
    temperature = 5
    y_target = sharpen(teacher_output, temperature)
    L_soft = nn.KLDivLoss()(F.log_softmax(student_output/temperature, dim=1), y_target)
    L_hard = F.cross_entropy(student_output, labels)
    sharp_KD_loss = alpha * L_soft + (1.-alpha) * L_hard
    return sharp_KD_loss

def loss_KL(student_output, teacher_output, T=5):
    temperature = T
    y_target = F.softmax(teacher_output/temperature, dim=1)
    y_student = F.log_softmax(student_output/temperature, dim=1)
    KL_div = nn.KLDivLoss()(y_student,y_target)
    return KL_div.mul(T**2)

def Loss_MMD():
    mmd_loss = nn.MSELoss(reduction="none")

def importance_loss(scores):
    Impi = scores.float().sum(0)
    l_imp = Impi.float().var() / (Impi.float().mean() ** 2 + 1e-10)
    return l_imp