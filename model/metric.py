#################################################################
# Code written by Sajad Darabi (sajad.darabi@cs.ucla.edu)
# For bug report, please contact author using the email address
#################################################################

import torch


def recall_k(output, target, mask, k=10, window=1):  # 所谓skip-gram的设定，完全靠metric来实现，model本身其实极其简单
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    for i in range(window):
        mi = mask[i + 1:] * mask[:-i - 1]  # 每次visit预测下面第i个visit，每个人预测自己的，这个逻辑上有点扯
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))  # 左右加一些0
        tm = mi[:-i - 1]  # 预测目标
        im = mi[i + 1:]  # 输入

        target_mask = torch.masked_select(idx, tm)  # 选择不被mask的预测对象visit的index
        input_mask = torch.masked_select(idx, im)  # 选择不被mask的预测对象visit的index
        #ii = ii.long()
        output = output[input_mask, :]  # 输出的visit code?
        output = output.float()
        target = target[target_mask, :]  # 正确的visit code
        target = target.float()

        _, tk = torch.topk(output, k)  # 输出的visit code的top k
        tt = torch.gather(target, 1, tk)  # 这个就不太清楚了
        r = torch.mean(torch.sum(tt, dim=1) / torch.sum(target, dim=1))
        if r != r:
            r = 0
    return r
