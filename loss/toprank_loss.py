import torch


def toprank_loss(labels_batchsize1, labels_batchsize2, hashrepresentations_batchsize1, hashrepresentations_batchsize2):
    batch_size, bit = hashrepresentations_batchsize1.shape
    labels_batchsize1 = labels_batchsize1 / torch.sqrt(torch.sum(torch.pow(labels_batchsize1, 2), 1)).unsqueeze(1)
    labels_batchsize2 = labels_batchsize2 / torch.sqrt(torch.sum(torch.pow(labels_batchsize2, 2), 1)).unsqueeze(1)

    labelsSimilarity = torch.matmul(labels_batchsize1, labels_batchsize2.t())  # [0,1]
    hashrepresentationsSimilarity = 1 / 2 * (bit -
                                             torch.matmul(hashrepresentations_batchsize1,
                                                          hashrepresentations_batchsize2.t()))  # [0,1]

    s1 = labelsSimilarity.repeat(1, batch_size)
    sij = s1.view(batch_size * batch_size, batch_size)
    s2 = labelsSimilarity.repeat(batch_size, 1)
    s3 = s2.view(batch_size, batch_size * batch_size)
    sik = s3.t()
    srank = sij - sik

    h1 = hashrepresentationsSimilarity.repeat(1, batch_size)
    hij = h1.view(batch_size * batch_size, batch_size)
    h2 = hashrepresentationsSimilarity.repeat(batch_size, 1)
    h3 = h2.view(batch_size, batch_size * batch_size)
    hik = h3.t()
    hrank = torch.sigmoid(hij - hik)

    rank = (srank - hrank > 0).type(torch.cuda.FloatTensor)

    rankloss = torch.sum(torch.pow(rank, 3)) / (batch_size * batch_size * batch_size)
    return rankloss