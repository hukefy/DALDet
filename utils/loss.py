
import torch
import torch.nn as nn

from utils.metrics import bbox_iou
# from utils.metrics import bbox_iou, bbox_union_minus_inter
from utils.torch_utils import de_parallel
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           cv2, segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()



class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        # true=1 p_t=pred_prob    true=0 p_t=1-pred_prob
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob) # p_t
        # true=1 alpha_factor=self.alpha    true=0 alpha_factor=1-self.alpha
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha) # alpha_t
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        MSE = nn.MSELoss(reduce=True, size_average=True)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets # 1.0, 0.0

        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            # BCEcls, BCEobj = QFocalLoss(BCEcls, g), QFocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.MSE = MSE
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device



    def __call__(self, p, targets, imgs, depths, pads):  # predictions, targets

        img_h = imgs.shape[-2]
        img_w = imgs.shape[-1]

        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        lmse = torch.zeros(1, device=self.device)  # box mse loss

        tcls, tbox, indices, anchors, tbox_ori = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            num = b.shape[0]

            n = b.shape[0]  # number of targets
            if n:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                pxy = pxy.sigmoid() * 2 - 0.5
                # https://github.com/ultralytics/yolov3/issues/168
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss


                # Depth-guided loss
                txy_ori, twh_ori = tbox_ori[i].chunk(2, 1)
                txy, twh = tbox[i].chunk(2, 1)
                offsets = txy_ori - txy
                pxy_ori = pxy + offsets
                pbox_ori = torch.cat((pxy_ori, pwh), 1)
                stride = (img_w) / (pi.shape[-2])
                tbox_mapped = tbox_ori[i] * stride
                pbox_mapped = pbox_ori * stride

                pbox_xyxy = xywh2xyxy(pbox_mapped)
                tbox_xyxy = xywh2xyxy(tbox_mapped)

                for m in range(num):
                    current_pbox = pbox_xyxy[m]
                    current_tbox = tbox_xyxy[m]
                    current_pbox_x1, current_pbox_y1, current_pbox_x2, current_pbox_y2 = current_pbox.chunk(4, 0)
                    current_tbox_x1, current_tbox_y1, current_tbox_x2, current_tbox_y2 = current_tbox.chunk(4, 0)
                    depth_index = b[m]
                    current_depth = depths[depth_index]
                    current_pad = pads[depth_index]
                    pad_w = current_pad[0]
                    pad_h = current_pad[1]
                    left = pad_w
                    right = img_w - pad_w
                    top = pad_h
                    bottom = img_h - pad_h

                    current_pbox_x1_clamped = torch.round(current_pbox_x1.clamp(left, right)).type(torch.LongTensor)
                    current_pbox_y1_clamped = torch.round(current_pbox_y1.clamp(top, bottom)).type(torch.LongTensor)
                    current_pbox_x2_clamped = torch.round(current_pbox_x2.clamp(left, right)).type(torch.LongTensor)
                    current_pbox_y2_clamped = torch.round(current_pbox_y2.clamp(top, bottom)).type(torch.LongTensor)
                    current_tbox_x1_clamped = torch.round(current_tbox_x1.clamp(left, right)).type(torch.LongTensor)
                    current_tbox_y1_clamped = torch.round(current_tbox_y1.clamp(top, bottom)).type(torch.LongTensor)
                    current_tbox_x2_clamped = torch.round(current_tbox_x2.clamp(left, right)).type(torch.LongTensor)
                    current_tbox_y2_clamped = torch.round(current_tbox_y2.clamp(top, bottom)).type(torch.LongTensor)

                    minc_x = torch.min(current_pbox_x1_clamped, current_tbox_x1_clamped)
                    minc_y = torch.min(current_pbox_y1_clamped, current_tbox_y1_clamped)
                    maxc_x = torch.max(current_pbox_x2_clamped, current_tbox_x2_clamped)
                    maxc_y = torch.max(current_pbox_y2_clamped, current_tbox_y2_clamped)

                    if (minc_x < maxc_x) and (minc_y < maxc_y):
                        MBR = current_depth[:, minc_y:maxc_y, minc_x:maxc_x]

                        pbox_MBR = MBR.clone()
                        transformed_pbox_x1 = current_pbox_x1_clamped - minc_x
                        transformed_pbox_y1 = current_pbox_y1_clamped - minc_y
                        transformed_pbox_x2 = current_pbox_x2_clamped - minc_x
                        transformed_pbox_y2 = current_pbox_y2_clamped - minc_y
                        pbox_MBR[:, transformed_pbox_y1:transformed_pbox_y2, transformed_pbox_x1:transformed_pbox_x2] = 0

                        tbox_MBR = MBR.clone()
                        transformed_tbox_x1 = current_tbox_x1_clamped - minc_x
                        transformed_tbox_y1 = current_tbox_y1_clamped - minc_y
                        transformed_tbox_x2 = current_tbox_x2_clamped - minc_x
                        transformed_tbox_y2 = current_tbox_y2_clamped - minc_y
                        tbox_MBR[:, transformed_tbox_y1:transformed_tbox_y2, transformed_tbox_x1:transformed_tbox_x2] = 0

                        lmse += self.MSE(pbox_MBR, tbox_MBR)
                    else:
                        print('Error in MBR computation')
                        lmse += 0.0




                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou: # False
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1: # 1.0
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lmse *= self.hyp['mse']
        bs = tobj.shape[0]  # batch size

        return (lmse*lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls, lmse, lmse*lbox)).detach()



    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets

        tcls, tbox, indices, anch, tbox_ori = [], [], [], [], []

        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl): # self.nl: number of detection layers
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio

                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']

                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)
            a, (b, c) = a.long().view(-1), bc.long().T
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            # Append
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)

            tbox_ori.append(torch.cat((gxy, gwh), 1))

        return tcls, tbox, indices, anch, tbox_ori
