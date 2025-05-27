from torch import nn
# from utils.utils import calc_map_k
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
from matplotlib import pyplot as plt

from utils.utils import calc_map_k


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def pr_curve(qB, rB, query_label, retrieval_label):
    num_query = qB.shape[0]
    num_bit = qB.shape[1]
    P = torch.zeros(num_query, num_bit + 1)
    R = torch.zeros(num_query, num_bit + 1)
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask
    return P, R


def p_topK(qB, rB, query_label, retrieval_label, K):
    num_query = query_label.shape[0]
    p = [0] * len(K)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm)[1][:total]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p


def PR_curve(precision, recall, epoch, flag):
    plt.figure()
    plt.title(flag + ' PR')
    plt.xlabel("recall")
    plt.ylabel("precision")
    index = [i for i in range(len(precision)) if precision[i] != 0]
    plt.plot(recall[index], precision[index], label=epoch, marker='o')
    plt.grid()
    plt.legend()
    plt.savefig('f1/' + str(epoch) + ' ' + flag + ' PR.png', bbox_inches='tight')
    plt.close()


K = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]


def P_topK(pk, epoch, flag2):
    plt.figure()
    plt.title(flag2 + ' TOP')
    plt.xlabel("N")
    plt.ylabel("precision")
    index = [i for i in range(len(pk)) if pk[i] != 0]

    plt.plot(K, pk[index], label=epoch, marker='o')
    plt.grid()
    plt.legend()
    plt.savefig('f2/' + str(epoch) + ' ' + flag2 + ' PTOP.png', bbox_inches='tight')
    plt.close()


def valid(batch_size, bit, use_gpu, img_model: nn.Module, txt_model: nn.Module, dataset, epoch, return_hash=False):
    # get query img and txt binary code
    dataset.query()
    qB_img = get_img_code(batch_size, bit, use_gpu, img_model, dataset)
    qB_txt = get_txt_code(batch_size, bit, use_gpu, txt_model, dataset)
    query_label = dataset.get_all_label()
    # get retrieval img and txt binary code
    dataset.retrieval()
    rB_img = get_img_code(batch_size, bit, use_gpu, img_model, dataset)
    rB_txt = get_txt_code(batch_size, bit, use_gpu, txt_model, dataset)
    retrieval_label = dataset.get_all_label()
    mAPi2t = calc_map_k(qB_img, rB_txt, query_label, retrieval_label)
    mAPt2i = calc_map_k(qB_txt, rB_img, query_label, retrieval_label)

    pk = p_topK(qB_img, rB_txt, query_label, retrieval_label, K)
    P_topK(pk.cpu().numpy(), epoch, 'i2t')
    pk = p_topK(qB_txt, rB_img, query_label, retrieval_label, K)
    P_topK(pk.cpu().numpy(), epoch, 't2i')

    precision, recall = pr_curve(qB_img, rB_txt, query_label, retrieval_label)
    PR_curve(precision.cpu().numpy(), recall.cpu().numpy(), epoch, 'i2t')

    precision, recall = pr_curve(qB_txt, rB_img, query_label, retrieval_label)
    PR_curve(precision.cpu().numpy(), recall.cpu().numpy(), epoch, 't2i')

    if return_hash:
        return mAPi2t, mAPt2i, qB_img.cpu(), qB_txt.cpu(), rB_img.cpu(), rB_txt.cpu(), query_label, retrieval_label
    return mAPi2t, mAPt2i


def get_img_code(batch_size, bit, use_gpu, img_model: nn.Module, dataset, isPrint=False):
    dataset.img_load()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, drop_last=True, pin_memory=True)
    # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    B_img = torch.zeros(len(dataset), bit, dtype=torch.float)
    if use_gpu:
        B_img = B_img.cuda()
    for data in tqdm(dataloader):
        index = data['index'].numpy()  # type: np.ndarray
        img = data['img']  # type: torch.Tensor
        if use_gpu:
            img = img.cuda()
        f = img_model(img)
        B_img[index, :] = f.data
        if isPrint:
            print(B_img[index, :])
    B_img = torch.sign(B_img)
    return B_img.cpu()


def get_txt_code(batch_size, bit, use_gpu, txt_model: nn.Module, dataset, isPrint=False):
    dataset.txt_load()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=8, drop_last=True, pin_memory=True)
    # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
    B_txt = torch.zeros(len(dataset), bit, dtype=torch.float)
    if use_gpu:
        B_txt = B_txt.cuda()
    for data in tqdm(dataloader):
        index = data['index'].numpy()  # type: np.ndarray
        txt = data['txt']  # type: torch.Tensor
        txt = txt.float()

        if use_gpu:
            txt = txt.cuda()
        g = txt_model(txt)
        B_txt[index, :] = g.data
        if isPrint:
            print(B_txt[index, :])
    B_txt = torch.sign(B_txt)
    return B_txt.cpu()
