#################################################################
# Code written by Sajad Darabi (sajad.darabi@cs.ucla.edu)
# For bug report, please contact author using the email address
#################################################################

import torch


def recall_k(output, target, mask, k=10, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    for i in range(1, window+1):
        mask_len = mask.shape[0]
        _shape = list(mask.shape)
        _shape[0] = mask_len - i
        maski = torch.ones(_shape).to(mask.device)
        for j in range(0, i + 1):
            maski = maski * mask[i - j:mask_len - j]  # 滑动mask
        mi = maski
        """
        每次visit预测旁边第i个visit(里面的code)，每个人预测自己的visit，这个逻辑上并不合理
        两次visit之间又不一定有关系,直觉上来说,大概率没关系
        """
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))  # 左右加一些0
        tm = mi[:-i]  # 预测目标 mask
        im = mi[i]  # 输入 mask

        target_mask = torch.masked_select(idx, tm)  # 选择不被mask的预测对象visit的index
        input_mask = torch.masked_select(idx, im)  # 选择不被mask的预测对象visit的index
        #ii = ii.long()
        """
        每个visit都是一个one-hot编码的向量，所以直接编码了一次visit里出现的所有code，也意味着多次发生的code只会记一次
        """
        output = output[input_mask, :]  # 输出的visit code
        output = output.float()
        target = target[target_mask, :]  # 正确的visit code
        target = target.float()

        _, tk = torch.topk(output, k)  # 输出的visit code的top k
        tt = torch.gather(target, 1, tk)  # topk 在target中的值,既topk的预测，在实际target中是1还是0
        r = torch.mean(torch.sum(tt, dim=1) / torch.sum(target, dim=1))  # 实际上topk命中target中出现的code的比例
        if r != r:  # TODO 这是什么鬼？
            r = 0
    return r  # TODO 按照这个逻辑，根本就不用for循环，直接返回最后一次的结果就行了，r并没有叠加
