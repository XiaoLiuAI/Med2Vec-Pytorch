import torch


def med2vec_loss(inputs, mask, probits, bce_loss, emb_w, ivec, jvec, window=1, eps=1.0e-8):
    """ returns the med2vec loss
    skip-gram的设定，完全靠metric来实现，model本身其实极其简单
    """
    def visit_loss(x, mask, probits, window=1):
        """
        x以visit为单位，mask标记patient之间的间隔
        :param x:
        :param mask:
        :param probits:
        :param window:
        :return:
        """
        loss = 0
        for i in range(1, window+1):
            if loss != loss:  # TODO 这是什么鬼？
                import pdb; pdb.set_trace()
            l = mask.shape[0]
            _shape = list(mask.shape)
            _shape[0] = l-i
            maski = torch.ones(_shape).to(mask.device)
            for j in range(0, i+1):
                maski = maski * mask[i-j:l-j]  # 滑动mask
            backward_preds = probits[i:] * maski  # 向后预测的目标
            forward_preds = probits[:-i] * maski  # 向前预测的目标
            #
            loss += bce_loss(forward_preds, x[i:].float()) + bce_loss(backward_preds, x[:-i].float())
        return loss

    def code_loss(emb_w, ivec, jvec, eps=1.e-6):
        """
        :param emb_w: 医学概念向量
        :param ivec: 同一个visit里面的i，j vector, 既一次admission/visit中的code对，zip(ivec, jvec) = list of pair of codes
        :param jvec:
        :param eps:
        :return:
        """
        norm = torch.sum(torch.exp(torch.mm(emb_w.t(), emb_w)), dim=1)  # normalize embedding

        # i与j之间的相似度，出现在同一个visit之中的code要相似?
        cost = -torch.log((torch.exp(torch.sum(emb_w[:, ivec].t() * emb_w[:, jvec].t(), dim=1)) / norm[ivec]) + eps)
        cost = torch.mean(cost)
        return cost

    vl = visit_loss(inputs, mask, probits, window=window)
    cl = code_loss(emb_w, ivec, jvec, eps=1.e-6)
    return {'visit_loss': vl, 'code_loss': cl}
