# This dataset is downloaded from https://zenodo.org/records/13126114

import json
from glob import glob
import os
from PIL import Image
import numpy as np
import nibabel as nib

from .syntra_data_converter import SyntraDataConverter

color_map = {
    "left_ventricle": [255, 0, 0 ],
    "right_ventricle": [0, 255, 0 ],
    "myocardium": [0, 0, 255]
}




class ACDCConverter(SyntraDataConverter):
    def __init__(self, syntra_data_path, customized_data_path, color_map, fg_ratio=0.05, img_size=384, month='september', **kwargs):
        """
        Note: This dataset is downloaded from https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb. 
        Since the foreground pixels are mostly in the center of the images, we set a low fg_ratio to better sample the patches.
        We follow https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9153905 to crop the images to 120x120 for training and testing.
        We only use the end-diastolic and end-systolic frames for segmentation. Besides, for few-shot data selection, we try to cover all the 
        pathological cases in the training data.

        Data structure in customized_data_path should be like:

        ├── testing
        │   ├── MANDATORY_CITATION.md
        │   ├── patient101
        │   │   ├── Info.cfg
        │   │   ├── MANDATORY_CITATION.md
        │   │   ├── patient101_4d.nii.gz
        │   │   ├── patient101_frame01_gt.nii.gz
        │   │   ├── patient101_frame01.nii.gz
        │   │   ├── patient101_frame14_gt.nii.gz
        │   │   └── patient101_frame14.nii.gz
        │   ├── patient102
        │   ├── ...
        ├── training
        │   ├── MANDATORY_CITATION.md
        │   ├── patient001
        │   │   ├── Info.cfg
        │   │   ├── MANDATORY_CITATION.md
        │   │   ├── patient001_4d.nii.gz
        │   │   ├── patient001_frame01_gt.nii.gz
        │   │   ├── patient001_frame01.nii.gz
        │   │   ├── patient001_frame10_gt.nii.gz
        │   │   └── patient001_frame10.nii.gz
        │   ├── patient002
        │   ├── ...

        """
        # use the scene flow data folder as input
        super().__init__(syntra_data_path, 
                        color_map=color_map, 
                        customized_data_path=customized_data_path, 
                        fg_ratio=fg_ratio,
                        syntra_img_size=img_size,
                        **kwargs)
        self.crop_size = 120
        self._parse_input_data()

    def _parse_input_data(self):
        self.meta = {}
        for split in ['train',  'test']:
            self.meta[split] = {}
            patients = os.listdir(os.path.join(self.customized_data_path, f"{split}ing"))
            patients = [p for p in patients if os.path.isdir(os.path.join(self.customized_data_path, f"{split}ing", p))]
            for patient in patients:
                self.meta[split][patient] = {}
                patient_folder = os.path.join(self.customized_data_path, f"{split}ing", patient)
                with open(os.path.join(patient_folder, 'Info.cfg'), 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    if 'ED' in line:
                        ed_frame = line.strip().split(':')[-1].strip()
                        self.meta[split][patient]['ED'] = ed_frame
                    elif 'ES' in line:
                        es_frame = line.strip().split(':')[-1].strip()
                        self.meta[split][patient]['ES'] = es_frame
                    elif 'Group' in line:
                        group = line.strip().split(':')[-1].strip()
                        self.meta[split][patient]['Group'] = group
    
    def _parse_base_name(self, img_file):
        base_name = os.path.basename(img_file).replace('.png', '')
        zone = img_file.split(os.sep)[-2]
        return f"{zone}_{base_name}"
    
    def _get_id_map(self):
        return {i:i for i in range(1, len(self.color_map)+1)}
    
    def load_nii(self, img_path, normalize=False):
        """
        Function to load a 'nii' or 'nii.gz' file, The function returns
        everyting needed to save another 'nii' or 'nii.gz'
        in the same dimensional space, i.e. the affine matrix and the header

        Parameters
        ----------

        img_path: string
        String with the path of the 'nii' or 'nii.gz' image file name.

        Returns
        -------
        Three element, the first is a numpy array of the image values,
        the second is the affine transformation of the image, and the
        last one is the header of the image.
        """
        nimg = nib.load(img_path)
        data = nimg.get_fdata()
        if normalize:
            data = (data - data.min()) / (data.max() - data.min())
        return data, nimg.affine, nimg.header

    def convert(self, split_list=['train', 'test']):
        self._prepare_convertion()

        # load processed data info if exists
        try:
            data_info = self.load_data_info()
        except:
            data_info = {}
        id_to_train_id = self._get_id_map()

        for split in split_list:
            if split not in ['train', 'test']:
                continue
            print(f"Processing split: {split} data")
            cur_meta = self.meta[split]
            split_samples = []
 
            for patient, info in cur_meta.items():
                patient_folder = os.path.join(self.customized_data_path, f"{split}ing", patient)
                for phase in ['ED', 'ES']:
                    frame_id = int(info[phase])
                    img_file = os.path.join(patient_folder, f"{patient}_frame{frame_id:02d}.nii.gz")
                    lbl_file = os.path.join(patient_folder, f"{patient}_frame{frame_id:02d}_gt.nii.gz")

                    imgs = self.load_nii(img_file, normalize=True)[0].astype(np.float32)
                    lbls = self.load_nii(lbl_file)[0].astype(np.uint8)

                    for i, (img, lbl) in enumerate(zip(imgs.transpose(2,0,1), lbls.transpose(2,0,1))):
                        non_zero_inds = np.nonzero(lbl)
                        if len(non_zero_inds[0]) == 0:
                            lbl_fg_ctr = np.array([lbl.shape[0]//2, lbl.shape[1]//2])
                        else:
                            lbl_fg_ctr = np.array(non_zero_inds).mean(axis=1).astype(np.int32)

                        img_pil = Image.fromarray((img * 255).astype(np.uint8))
                        lbl_pil = Image.fromarray(lbl)
                        filename = f'{patient}_{phase}_frame{frame_id:02d}_slice{i:03d}'
                        sample_full_name = f"{self.dataset_name}.{filename}"
                        split_samples.append(sample_full_name)
                        
                        img_pil = img_pil.crop((lbl_fg_ctr[1] - self.crop_size//2,
                                               lbl_fg_ctr[0] - self.crop_size//2,
                                               lbl_fg_ctr[1] + self.crop_size//2,
                                               lbl_fg_ctr[0] + self.crop_size//2))
                        lbl_pil = lbl_pil.crop((lbl_fg_ctr[1] - self.crop_size//2,
                                               lbl_fg_ctr[0] - self.crop_size//2,
                                               lbl_fg_ctr[1] + self.crop_size//2,
                                               lbl_fg_ctr[0] + self.crop_size//2))
                        lbl_pil = self._convert_annotation(np.array(lbl_pil), id_to_train_id)
                        img_save_path = os.path.join(self.img_dir, f"{filename}.png")
                        lbl_save_path = os.path.join(self.lbl_dir, f"{filename}.png")
                        img_pil.save(img_save_path)
                        lbl_pil.save(lbl_save_path)
                        data_info[sample_full_name] = self._parse_label_ratios(np.array(lbl_pil))
                        

            self.save_samples(split_samples, split)
            self.save_data_info(data_info) # save at finishing every split to avoid data loss



    



