import numpy as np
import torch

from ..tools.eval import Evaluator


class Validator:
    """Score a model on a held-out pairing file, the way `test.py` scores it.

    This used to call the model as `model(img_src, lbl_src, img_tgt)` straight off the
    dataloader, without the `prepare_inputs` step the efficient dataset needs, which reached the
    prompt encoder with a mask of the wrong rank and raised on the first validation. It also
    looped over a hard-coded `{dataset: [label ids]}` table without using the label for
    anything - so each dataset was evaluated three or four times over - and returned `ious[9]`,
    an index that happens to be the mean only while the corpus has exactly nine dataset-label
    combinations.

    The loop below is the one in `test.py`: same call, same squeeze, same per-class accounting
    through the evaluator's `meta`, so a validation number and a test number mean the same thing.
    """

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.device = device

    def validate(self, model):
        evaluator = Evaluator()
        model.eval()
        with torch.no_grad():
            for data in self.dataloader:
                img_src, lbl_src, img_tgt, lbl_tgt = [d.to(self.device) for d in data[:4]]
                meta = data[4] if len(data) > 4 else None
                if hasattr(self.dataset, "prepare_inputs"):
                    img_src, lbl_src, img_tgt, lbl_tgt, meta = self.dataset.prepare_inputs(
                        img_src, lbl_src, img_tgt, lbl_tgt, meta)
                pred = model(img_src, lbl_src, img_tgt).sigmoid()
                evaluator.add_sample(pred.squeeze(2), lbl_tgt, meta=meta)
        model.train()

        per_class = evaluator.eval()
        names = sorted(per_class)
        ious = [float(per_class[n][0]) for n in names]
        mean_iou = float(np.mean(ious)) if ious else 0.0
        rows = [
            ("IoU", ious),
            ("Prec", [float(per_class[n][1]) for n in names]),
            ("Rec", [float(per_class[n][2]) for n in names]),
        ]
        return mean_iou, per_class, self.draw_txt_table(names + ["mean"], rows, mean_iou)

    def draw_txt_table(self, header, rows, mean_iou, col_width=12):
        lines = [["Data"] + list(header)]
        for name, values in rows:
            summary = np.mean(values) if values else 0.0
            lines.append([name] + [f"{v * 100:.2f}" for v in values] + [f"{summary * 100:.2f}"])
        border = "+" + "+".join(["-" * col_width] * len(lines[0])) + "+"
        table = border + "\n"
        for row in lines:
            table += "|" + "|".join(str(c).center(col_width) for c in row) + "|\n" + border + "\n"
        return table
