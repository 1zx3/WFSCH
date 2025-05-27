import os
import time

from training import MESDCH

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    datasets = ['mirflickr25k']
    bits = [16, 32, 64]
    for ds in datasets:
        for bit in bits:
            MESDCH.train(ds, bit, batch_size=64, issave=False, max_epoch=150, use_gpu=True)

    print(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

