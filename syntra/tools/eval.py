import numpy as np
import torch


class Evaluator:
    def __init__(self):
        self.tp = {}
        self.union = {}
        self.pos_pred = {}
        self.pos_gt = {}
        self.overall_tp = {} # including background
        self.n = {}
        self.notions = []
    
    def _add_new_notions(self, notion_id):
        if isinstance(notion_id, int):
            notion_id = f"notion_{notion_id}"
        self.tp[notion_id] = 0
        self.union[notion_id] = 0
        self.pos_pred[notion_id] = 0
        self.pos_gt[notion_id] = 0
        self.overall_tp[notion_id] = 0
        self.n[notion_id] = 0
        self.notions.append(notion_id)
    
    def _parse_notion_id(self, cls_full_name):
        if isinstance(cls_full_name, list):
            return [self._parse_notion_id(x) for x in cls_full_name]
        sub_dataset_name, cls_name = cls_full_name.split('/')
        dataset_name= sub_dataset_name.split('.')[0]
        return f"{dataset_name}.{cls_name}"

    
    def _add_one_sample(self, pred_cls, target, notion_id):
        if isinstance(notion_id, list):
            for p, t, n_id in zip(pred_cls, target, notion_id):
                self._add_one_sample(p, t, n_id)
        else:
            assert isinstance(notion_id, str)
            if notion_id not in self.notions:
                    self._add_new_notions(notion_id)
            # iou
            tp = torch.logical_and(pred_cls, target)
            union = torch.logical_or(pred_cls, target)
            self.n[notion_id] += 1
            self.overall_tp[notion_id] += (pred_cls == target).sum() / target.numel()
            self.tp[notion_id] += torch.sum(tp)
            self.union[notion_id] += torch.sum(union)
            self.pos_pred[notion_id] += torch.sum(pred_cls)
            self.pos_gt[notion_id] += torch.sum(target)

    def add_sample(self, pred, target, thresh=0.5, meta=None):
        assert pred.shape == target.shape
        B, N = target.shape[:2]

        if meta is None:
            notion_ids = [[f"notion_{len(self.notions) + 1}"]] * B
        else:
            assert len(meta) == B
            assert 'cls_name' in meta[0]
            notion_ids = [self._parse_notion_id(x['cls_name']) for x in meta]

        pred_cls = pred > thresh

        for args in zip(pred_cls, target, notion_ids):
            self._add_one_sample(*args)

    def eval(self):
        res = {}
        for notion_id in self.notions:
            iou = self.tp[notion_id] / self.union[notion_id] if self.union[notion_id] > 0 else 0
            prec = self.tp[notion_id] / self.pos_pred[notion_id] if self.pos_pred[notion_id] > 0 else 0
            rec = self.tp[notion_id] / self.pos_gt[notion_id] if self.pos_gt[notion_id] > 0 else 0
            acc = self.overall_tp[notion_id] / self.n[notion_id] if self.n[notion_id] > 0 else 0
            res[notion_id] = (iou, prec, rec, acc)

        return res


def cal_all(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection.sum() / union.sum()
    pos_pred = pred_mask.sum()
    pos_gt = gt_mask.sum()
    prec = intersection / pos_pred
    rec = intersection / pos_gt
    return iou, prec, rec

