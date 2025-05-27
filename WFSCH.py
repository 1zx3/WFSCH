from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
from models.alexnet import alexnet
from loss.toprank_loss import toprank_loss
from models import Txt_net
from models.resnet import resnet50, resnet101
from utils.valid import valid
from loss.multisimilarity import multilabelsimilarityloss_KL
from loss.quantizationloss import quantizationLoss
from utils.save_results import save_hashcodes
import os

from models.vscm import Semantic_Match
from datasets.mirflckr25k import get_single_datasets

def calc_neighbor(label1, label2, use_gpu):
    # calculate the similar matrix
    if use_gpu:
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
    else:
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.FloatTensor)
    return Sim


def train(dataset_name: str, bit: int, issave=False, batch_size=64, use_gpu=True, max_epoch=150, lr=10 ** (-2),
          isvalid=True, alpha=0.7, beta=1.3, gamma=1.8, eta=0.01):
    print('datasetname = %s; bit = %d' % (dataset_name, bit))

    if dataset_name.lower() == 'mirflickr25k':
        from datasets.mirflckr25k import get_single_datasets
        tag_length = 2885
        label_length = 275
    else:
        raise ValueError("there is no datasets name is %s" % dataset_name)

    train_data, valid_data = get_single_datasets(batch_size=batch_size)

    sc_sa = Semantic_Match(bit)
    img_model = resnet50(bit)
#    img_model = alexnet(bit)
    txt_model = Txt_net(tag_length, bit)
    label_model = Txt_net(label_length, bit)

    if use_gpu:
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()
        label_model = label_model.cuda()
        sc_sa = sc_sa.cuda()

    num_train = len(train_data)
    train_L = train_data.get_all_label()

    F_buffer = torch.randn(num_train, bit)
    G_buffer = torch.randn(num_train, bit)
    L_buffer = torch.randn(num_train, bit)

    if use_gpu:
        train_L = train_L.cuda()
        F_buffer = F_buffer.cuda()
        G_buffer = G_buffer.cuda()
        L_buffer = L_buffer.cuda()

    B = torch.sign(F_buffer + G_buffer + L_buffer)

    optimizer_img = SGD(img_model.parameters(), lr=lr)
    optimizer_txt = SGD(txt_model.parameters(), lr=lr)
    optimizer_label = SGD(label_model.parameters(), lr=lr)

    learning_rate = np.linspace(lr, np.power(10, -6.), max_epoch + 1)
    result = {
        'loss': []
    }

    ones = torch.ones(batch_size, 1)  # 128*1
    ones_ = torch.ones(num_train - batch_size, 1)  # (10000-128)*1
    # unupdated_size = num_train - batch_size  # 10000-128

    max_mapi2t = max_mapt2i = 0.
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    for epoch in range(max_epoch):
        # train label net
        train_data.img_txt_load()  # 兩個都true
        train_data.re_random_item()
        for data in tqdm(train_loader):  # tqdm用来显示进度条的
            ind = data['index'].numpy()

            sample_L = data['label']
            sample_L_train = sample_L.unsqueeze(1).unsqueeze(-1).type(torch.float)
            image = data['img']
            text = data['txt']
            if use_gpu:
                text = text.cuda()
                image = image.cuda()
                sample_L = sample_L.cuda()
                ones = ones.cuda()  # 128*1
                ones_ = ones_.cuda()  # (10000-128)*1
                sample_L_train = sample_L_train.cuda()

            cur_g = txt_model(text)
            cur_f = img_model(image)
            cur_l = label_model(sample_L_train)  # cur_f: (batch_size, bit)

            # SCSA
            attention = sc_sa(cur_l)
            weighted_label = attention * sample_L.unsqueeze(2)
            sample_L = torch.sum(weighted_label, dim=2)

            L_buffer[ind, :] = cur_l.data  # F_buffer:10000*64
            F = Variable(F_buffer)
            G = Variable(G_buffer)
            L = Variable(L_buffer)

            ranklossll = toprank_loss(sample_L, sample_L, cur_l, cur_l)
            ranklosslx = toprank_loss(sample_L, sample_L, cur_l, cur_f)
            ranklossly = toprank_loss(sample_L, sample_L, cur_l, cur_g)

            KLloss_ll = beta * multilabelsimilarityloss_KL(sample_L, train_L, cur_l, L)
            KLloss_lx = alpha * multilabelsimilarityloss_KL(sample_L, train_L, cur_l, F)
            KLloss_ly = alpha * multilabelsimilarityloss_KL(sample_L, train_L, cur_l, G)
            quantization_l = gamma * quantizationLoss(cur_l, B[ind, :])
            # new_loss

            #S = calc_neighbor(sample_L, train_L, use_gpu)  # S: (batch_size, num_train)
            # W = W_[ind, :]
            # W_center = W
            # W_center = W_center.T[ind, :]
            # W_center = W_center.T
            # Smulti = Sc_[ind, :]
            # Smulti = Smulti.T[ind, :]
            # Smulti = Smulti.T
            # if use_gpu:
            #     S = S.cuda()
                # W = W.cuda()
                # W_center = W_center.cuda()
                # Smulti = Smulti.cuda()

            # calculate loss
#             theta_ll = 1.0 / 2 * torch.matmul(cur_l, L.t())
#             logloss_ll = -torch.sum(S * theta_ll - torch.log(1.0 + torch.exp(theta_ll)))
#             theta_lx = 1.0 / 2 * torch.matmul(cur_l, F.t())
#             logloss_lx = -torch.sum(S * theta_lx - torch.log(1.0 + torch.exp(theta_lx)))
#             theta_ly = 1.0 / 2 * torch.matmul(cur_l, G.t())
#             logloss_ly = -torch.sum(S * theta_ly - torch.log(1.0 + torch.exp(theta_ly)))

#             loss_l = logloss_lx + logloss_ly + logloss_ll

#             loss_l = loss_l / (num_train * batch_size)
            # multiloss_l1 = torch.sum((torch.pow((1.0 / bit * torch.matmul(torch.tanh(cur_l), torch.tanh(F[ind, :].t())) - Smulti), 2)))
            # multiloss_l2 = torch.sum((torch.pow((1.0 / bit * torch.matmul(torch.tanh(cur_l), torch.tanh(G[ind, :].t())) - Smulti), 2)))
            # multiloss_l = multiloss_l1 + multiloss_l2
            # loss_l += multiloss_l

            loss_l = KLloss_ll + KLloss_lx + KLloss_ly + quantization_l + eta * (ranklossll + ranklossly + ranklosslx)

            optimizer_label.zero_grad()
            loss_l.backward(retain_graph=True)
            optimizer_label.step()
        train_data.both_load()  # 兩個都false
        # train image net
        train_data.img_txt_load()  # 兩個都true
        train_data.re_random_item()
        for data in tqdm(train_loader):  # tqdm用来显示进度条的
            ind = data['index'].numpy()

            sample_L = data['label']
            sample_L_train = sample_L.unsqueeze(1).unsqueeze(-1).type(torch.float)

            image = data['img']
            text = data['txt']

            if use_gpu:
                image = image.cuda()
                text = text.cuda()

                sample_L = sample_L.cuda()
                ones = ones.cuda()  # 128*1
                ones_ = ones_.cuda()  # (10000-128)*1
                sample_L_train = sample_L_train.cuda()

            cur_f = img_model(image)
            cur_g = txt_model(text)

            cur_l = label_model(sample_L_train)  # cur_f: (batch_size, bit)

            attention = sc_sa(cur_f)
            weighted_label = attention * sample_L.unsqueeze(2)
            sample_L = torch.sum(weighted_label, dim=2)

            F_buffer[ind, :] = cur_f.data
            F = Variable(F_buffer)
            L = Variable(L_buffer)
            G = Variable(G_buffer)

            ranklossxx = toprank_loss(sample_L, sample_L, cur_f, cur_f)
            ranklossxl = toprank_loss(sample_L, sample_L, cur_f, cur_l)
            ranklossxy = toprank_loss(sample_L, sample_L, cur_f, cur_g)

            KLloss_xx = beta * multilabelsimilarityloss_KL(sample_L, train_L, cur_f, F)
            KLloss_xl = alpha * multilabelsimilarityloss_KL(sample_L, train_L, cur_f, L)
            quantization_x = gamma * quantizationLoss(cur_f, B[ind, :])
            # new loss
            # S = calc_neighbor(sample_L, train_L, use_gpu)
            # W = W_[ind,:]
            # Smulti = Sc_[ind, :]
            # Smulti = Smulti.T[ind, :]
            # Smulti = Smulti.T
            # unupdated_ind = np.setdiff1d(range(num_train), ind)
            LB = Variable(L_buffer)
#             if use_gpu:
#                 S = S.cuda()
#                 # W = W.cuda()
#                 # Smulti = Smulti.cuda()

#             theta_x = 1.0 / 2 * torch.matmul(cur_f, G.t())
#             logloss_x = -torch.sum(S * theta_x - torch.log(1.0 + torch.exp(theta_x)))
#             #            balance_x = torch.sum(torch.pow(cur_f.t().mm(ones) + F[unupdated_ind].t().mm(ones_), 2))

#             theta_xl = 1.0 / 2 * torch.matmul(cur_f, LB.t())
#             logloss_xl = -torch.sum(S * theta_xl - torch.log(1.0 + torch.exp(theta_xl)))
#             #            loss_x = logloss_x + eta * balance_x + logloss_xl
#             loss_x = logloss_x + logloss_xl

#             loss_x = loss_x / (batch_size * num_train)
            # multiloss_l1 = torch.sum((torch.pow((1.0 / bit * torch.matmul(torch.tanh(cur_l), torch.tanh(F[ind, :].t())) - Smulti), 2)))
            # multiloss_l2 = torch.sum((torch.pow((1.0 / bit * torch.matmul(torch.tanh(cur_l), torch.tanh(G[ind, :].t())) - Smulti), 2)))
            # multiloss_l = multiloss_l1 + multiloss_l2
            # loss_x += multiloss_l

            loss_x = KLloss_xx + KLloss_xl + quantization_x + eta * (ranklossxx + ranklossxl + ranklossxy)

            optimizer_img.zero_grad()
            loss_x.backward(retain_graph=True)
            optimizer_img.step()

        # train txt net
        train_data.img_txt_load()  # 兩個都true
        train_data.re_random_item()
        for data in tqdm(train_loader):  # tqdm用来显示进度条的
            ind = data['index'].numpy()

            sample_L = data['label']
            sample_L_train = sample_L.unsqueeze(1).unsqueeze(-1).type(torch.float)
            image = data['img']
            text = data['txt']
            if use_gpu:
                text = text.cuda()
                image = image.cuda()

                sample_L = sample_L.cuda()
                sample_L_train = sample_L_train.cuda()

            cur_g = txt_model(text)
            cur_f = img_model(image)

            cur_l = label_model(sample_L_train)

            attention = sc_sa(cur_g)
            weighted_label = attention * sample_L.unsqueeze(2)
            sample_L = torch.sum(weighted_label, dim=2)

            G_buffer[ind, :] = cur_g.data
            F = Variable(F_buffer)
            L = Variable(L_buffer)
            G = Variable(G_buffer)

            ranklossyy = toprank_loss(sample_L, sample_L, cur_g, cur_g)
            ranklossyl = toprank_loss(sample_L, sample_L, cur_g, cur_l)
            ranklossyx = toprank_loss(sample_L, sample_L, cur_g, cur_f)

            KLloss_yy = beta * multilabelsimilarityloss_KL(sample_L, train_L, cur_g, G)
            KLloss_yl = alpha * multilabelsimilarityloss_KL(sample_L, train_L, cur_g, L)
            quantization_y = gamma * quantizationLoss(cur_g, B[ind, :])

            # new loss
            # S = calc_neighbor(sample_L, train_L, use_gpu)
            # W = W_[ind,:]
            # Smulti = Sc_[ind, :]
            # Smulti = Smulti.T[ind, :]
            # Smulti = Smulti.T
#             unupdated_ind = np.setdiff1d(range(num_train), ind)
#             LB = Variable(L_buffer)
#             if use_gpu:
#                 S = S.cuda()
#                 # W = W.cuda()
#                 # Smulti = Smulti.cuda()

#             theta_y = 1.0 / 2 * torch.matmul(cur_g, F.t())
#             logloss_y = -torch.sum(S * theta_y - torch.log(1.0 + torch.exp(theta_y)))
#             #            balance_y = torch.sum(torch.pow(cur_g.t().mm(ones) + G[unupdated_ind].t().mm(ones_), 2))

#             theta_yl = 1.0 / 2 * torch.matmul(cur_g, LB.t())
#             logloss_yl = -torch.sum(S * theta_yl - torch.log(1.0 + torch.exp(theta_yl)))
#             #            loss_y = logloss_y + eta * balance_y + logloss_yl
#             loss_y = logloss_y + logloss_yl
#             loss_y = loss_y / (num_train * batch_size)
            # multiloss_l1 = torch.sum((torch.pow((1.0 / bit * torch.matmul(torch.tanh(cur_l), torch.tanh(F[ind, :].t())) - Smulti), 2)))
            # multiloss_l2 = torch.sum((torch.pow((1.0 / bit * torch.matmul(torch.tanh(cur_l), torch.tanh(G[ind, :].t())) - Smulti), 2)))
            # multiloss_l = multiloss_l1 + multiloss_l2
            # loss_y += multiloss_l

            loss_y =  KLloss_yy + KLloss_yl + quantization_y + eta * (ranklossyy + ranklossyl + ranklossyx)

            optimizer_txt.zero_grad()
            loss_y.backward(retain_graph=True)
            optimizer_txt.step()

        print('...epoch: %3d, LabelLoss: %3.3f, ImgLoss:%3.3f,TxtLoss:%3.3f,lr: %f' % (
            epoch + 1, loss_l, loss_x, loss_y, lr))

        # update B
        B = torch.sign(F_buffer + G_buffer + L_buffer)

        if isvalid:
            mapi2t, mapt2i = valid(batch_size, bit, use_gpu, img_model, txt_model, valid_data, epoch+1)
            if mapi2t >= max_mapi2t:
                max_mapi2t = mapi2t
            if mapt2i >= max_mapt2i:
                max_mapt2i = mapt2i
            with open("calc16.txt", 'a', encoding='utf-8') as f:
                f.write(f"{epoch}: mapi2t:{str(mapi2t.item())} mapt2i: {str(mapt2i.item())} \n")
            print(
                '...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f;MAX_MAP(i->t): %3.4f, MAX_MAP(t->i): %3.4f' % (
                    epoch + 1, mapi2t, mapt2i, max_mapi2t, max_mapt2i))

        lr = learning_rate[epoch + 1]

        # set learning rate
        for param in optimizer_label.param_groups:
            param['lr'] = lr
        for param in optimizer_img.param_groups:
            param['lr'] = lr
        for param in optimizer_txt.param_groups:
            param['lr'] = lr

    print('...training procedure finish')
    if isvalid:
        print('max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
        if issave:
            save_hashcodes(batch_size, use_gpu, bit, img_model, txt_model, dataset_name, valid_data, 'MESDCH')
    else:
        mapi2t, mapt2i = valid(batch_size, bit, use_gpu, img_model, txt_model, valid_data)
        print('max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (mapi2t, mapt2i))


