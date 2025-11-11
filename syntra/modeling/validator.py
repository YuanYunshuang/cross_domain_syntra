import copy
import torch
import numpy as np

from ..tools.eval import Evaluator


class Validator:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.dataset = self.dataloader.dataset
        self.device = device
        self.data_list = copy.deepcopy(self.dataset.file_list)
        self.dataset_list = {'hameln': [1, 2, 3, 4],
                             'bayern': [1, 2, 4],
                             'siegfried.railway': [1],
                             'siegfried.vineyard': [2]}
        self.data_dict = {x: [] for x in self.dataset_list.keys()}

        for ds in self.dataset_list.keys():
            for d in self.data_list:
                if ds in d:
                    self.data_dict[ds].append(d)

    def validate(self, model):
        results = {}
        for dataset, cur_samples in self.data_dict.items():
            self.dataset.file_list = cur_samples
            results[dataset] = {}
            for label in self.dataset_list[dataset]:
                evaluator = Evaluator()
                with (torch.no_grad()):
                    for step, data in enumerate(self.dataloader):
                        img_src, lbl_src, img_tgt, lbl_tgt = [d.to(self.device) for d in data[:4]]
                        mapsam_pred = model(img_src, lbl_src, img_tgt)
                        lbl_pred = mapsam_pred.sigmoid()
                        evaluator.add_sample(lbl_pred.squeeze(1), lbl_tgt)

                avg_eval = evaluator.eval()
                results[dataset][label] = avg_eval

        ious = []
        precisions = []
        recalls = []
        for dataset, result in results.items():
            for label, res in result.items():
                ious.append(res[0].item())
                precisions.append(res[1].item())
                recalls.append(res[2].item())
        ious.append(np.mean(ious))
        precisions.append(np.mean(precisions))
        recalls.append(np.mean(recalls))

        results = [ious, precisions, recalls]
        results_str = self.draw_txt_table(
            ['H1', 'H2', 'H3', 'H4', 'D1', 'D2', 'D4', 'RW', 'VY', 'mean'],
            results,
        )
        return ious[9], results, results_str

    def draw_txt_table(self, header, results, col_width=10):
        num_columns = 11

        rows = [header]
        for res in results:
            rows.append([f"{x * 100:.2f}" for x in res])

        for i, k in enumerate(['Data', 'IoU', 'Prec', 'Rec']):
            rows[i] = [k] + rows[i]

        # Create horizontal border
        horizontal_border = '+' + '+'.join(['-' * col_width] * num_columns) + '+'

        # Build the table
        table = horizontal_border + '\n'
        for row in rows:
            row_line = '|'.join(cell.center(col_width) for cell in row)
            table += f"|{row_line}|\n"
            table += horizontal_border + '\n'
        return table
