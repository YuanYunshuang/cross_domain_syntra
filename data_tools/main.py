
import os
from converting.forest_monitoring_trees import ForestMonitoringTreesConverter
from converting.syntra_data_converter import SyntraDataConverter


def generate_for_one_dataset(converter_name, input_path, output_path, tasks:list, fg_ratio:float=0.05):
    converter_cls = globals().get(converter_name, None)
    converter = converter_cls(
        syntra_data_path=output_path,
        customized_data_path=input_path,
        fg_ratio=fg_ratio
    )

    for t in tasks:
        print('*' * 30 + f"\nStarting task: {t['task']} with args: {t.get('args', {})}")
        assert hasattr(converter, t['task']), f"Unknown task: {t}"
        getattr(converter, t['task'])(**t['args'])
        print(f"Done. \n" + '*' * 30)


def generate_pairing_for_one_dataset(
    output_path,
    src_info_file,
    tgt_info_file,
    fg_ratio=0.001,
):
    converter = SyntraDataConverter(
        syntra_data_path=output_path,
        fg_ratio=fg_ratio
    )

    converter.pairing_intra_cls(
        src_info_file=src_info_file,
        tgt_info_file=tgt_info_file,
    )


def convert_cityscapes(input_path, output_path, fg_ratio=0.05):
    from converting.cityscapes import CityScapesConverter, color_map
    converter = CityScapesConverter(
        syntra_data_path=output_path,
        customized_data_path=input_path,
        color_map=color_map,
        fg_ratio=fg_ratio
    )
    # converter.convert(['test', 'val'])
    # converter.select_cls_samples(n=100)
    pairing_options = [
        # ("cls_selection_100.json", "cls_selection_100.json"),
        # ("cls_selection_100.json", "test_samples.txt"),
        # ("train_samples.txt", "train_samples.txt"),
        ("train_samples.txt", "test_samples.txt"),
    ]
    for src_file, tgt_file in pairing_options:
        converter.pairing_intra_cls(src_info_file=src_file, tgt_info_file=tgt_file)


def convert_kitti(input_path, output_path, fg_ratio=0.05):
    from converting.kitti import KittiConverter, color_map
    converter = KittiConverter(
        syntra_data_path=output_path,
        customized_data_path=input_path,
        color_map=color_map,
        fg_ratio=fg_ratio
    )
    converter.convert(['train', 'test'])
    converter.select_cls_samples(n=100)
    pairing_options = [
        ("cls_selection_100.json", "cls_selection_100.json"),
        ("cls_selection_100.json", "test_samples.txt"),
        ("train_samples.txt", "train_samples.txt"),
        ("train_samples.txt", "test_samples.txt"),
    ]
    for src_file, tgt_file in pairing_options:
        converter.pairing_intra_cls(src_info_file=src_file, tgt_info_file=tgt_file)


def convert_demo(convert_obj, input_path, output_path, color_map, fg_ratio=0.05, **kwargs):
    converter = convert_obj(
        syntra_data_path=output_path,
        customized_data_path=input_path,
        color_map=color_map,
        fg_ratio=fg_ratio,
        **kwargs
    )

    converter.convert(['train', 'test', 'val'])
    # converter.select_cls_samples(n=100)
    # converter.select_cls_samples(n=50)
    # converter.select_cls_samples(n=10)
    # pairing_options = [
    #     # ("cls_selection_100.json", "cls_selection_100.json"),
    #     # ("cls_selection_100.json", "test_samples.txt"),
    #     # ("cls_selection_50.json", "cls_selection_50.json"),
    #     # ("cls_selection_50.json", "test_samples.txt"),
    #     # ("cls_selection_10.json", "cls_selection_10.json"),
    #     # ("cls_selection_10.json", "test_samples.txt"),
    #     ("train_samples.txt", "train_samples.txt"),
    #     ("train_samples.txt", "test_samples.txt"),
    # ]
    # for src_file, tgt_file in pairing_options:
    #     converter.pairing_sample_intra_cls(src_info_file=src_file, tgt_info_file=tgt_file)


def convert_siegfried(convert_obj, input_path, output_path, color_map, selections, fg_ratio=0.05, **kwargs):
    converter = convert_obj(
        syntra_data_path=output_path,
        customized_data_path=input_path,
        color_map=color_map,
        fg_ratio=fg_ratio,
        **kwargs
    )

    pairing_options = []

    # converter.convert(['train', 'test'])
    # pairing_options = [
    #     ("train_samples.txt", "train_samples.txt"),
    #     ("train_samples.txt", "test_samples.txt"),
    # ]
        
    for s in selections:
        if isinstance(s, int):
            converter.select_cls_samples(n=s)
            pairing_options.append((f"cls_selection_{s}.json", f"cls_selection_{s}.json"))
            pairing_options.append((f"cls_selection_{s}.json", "test_samples.txt"))
        else:
            converter.select_samples(s)
            pairing_options.append((f"{s}_samples.txt", f"{s}_samples.txt"))
            pairing_options.append((f"{s}_samples.txt", "test_samples.txt"))

    for src_file, tgt_file in pairing_options:
        converter.pairing_sample_intra_cls(src_info_file=src_file, tgt_info_file=tgt_file)

def convert_syntra2(input_path, output_path, subset, fg_ratio=0.05):
    from converting.syntra2 import Syntra2Converter
    converter = Syntra2Converter(
        syntra_data_path=output_path,
        customized_data_path=input_path,
        subset=subset,
        fg_ratio=fg_ratio
    )
    converter.convert(['train', 'test'])

    # pairing_options = [
    #     ("train_samples.txt", "train_samples.txt"),
    #     ("train_samples.txt", "test_samples.txt"),
    # ]

    # for n in [10, 50, 100]:
    #     converter.select_cls_samples(n=n)
    #     pairing_options.append((f"s{n}.json", f"s{n}.json"))
    #     pairing_options.append((f"s{n}.json", "test_samples.txt"))
    # for src_file, tgt_file in pairing_options:
    #     converter.pairing_intra_cls(src_info_file=src_file, tgt_info_file=tgt_file)


def update_lbls():
    from PIL import Image
    import numpy as np
    from glob import glob
    data_path = "/home/yuan/data/syntra"
    data_paths = glob(os.path.join(data_path, "**/color_map.json"), recursive=True)
    for dataset_path in data_paths:
        dataset_path = os.path.dirname(dataset_path)
        print(f"Processing dataset at {dataset_path}")
        converter = SyntraDataConverter(
            syntra_data_path=dataset_path,
            customized_data_path=None,
            fg_ratio=0.05
        )
        color_map_palette = None
        lbl_palette = None
        lbl_start_index = None
        update = False
        for i, lbl_file in enumerate(os.listdir(f"{dataset_path}/lbls")):
            lbl_file = os.path.join(dataset_path, 'lbls', lbl_file)
            lbl = Image.open(lbl_file)
            if i == 0:
                uniq_lbls = np.unique(np.array(lbl))
                color_map_palette = np.array(converter.palette)
                lbl_palette = np.array(lbl.getpalette())
                if 'railway' in converter.color_map:
                    lbl_start_index = 5  # railway is index 0
                elif 'vineyard' in converter.color_map:
                    lbl_start_index = 6  # vineyard is index 0
                else:
                    lbl_start_index = lbl_palette.reshape(-1, 3).sum(1).nonzero()[0][0]
                print("color map:", converter.palette[:30])
                print("lbl palette:", lbl.getpalette()[:30])
                print("unique labels in image:", uniq_lbls)
                print("label start index:", lbl_start_index)

                if (color_map_palette != lbl_palette).any():
                    update = True
                else:
                    print("Palettes match. No update needed.")
                    break
            if update:
                uniq_lbls = np.unique(np.array(lbl))
                # print('uniq_lbls before update:', uniq_lbls)
                if uniq_lbls.max() > len(converter.color_map):
                    lbl = np.array(lbl)
                    if 'bayern' in lbl_file:
                        lbl[lbl == 4] = 3
                    else:
                        for idx in range(len(converter.color_map)):
                            lbl[lbl == lbl_start_index + idx] = idx + 1
                    lbl = Image.fromarray(lbl, mode="P")
                
                lbl.putpalette(converter.palette)
                lbl.save(lbl_file)
                print(f"Updated palette for {lbl_file}")


if __name__ == "__main__":
    """
    Example usage for function generate_for_one_dataset:  
    
        >> input_directory = "/home/yuan/data/forest_monitoring/tree_data"
        >> output_directory = "/home/yuan/data/syntra/forest_monitoring/forest_monitoring.trees"
        >> fg_ratio = 0.05
        >> 
        >> generate_for_one_dataset(
        >>     converter_name="ForestMonitoringTreesConverter",
        >>     input_path=input_directory,
        >>     output_path=output_directory,
        >>     fg_ratio=fg_ratio,
        >>     tasks=[
        >>         {'task': 'convert', 'args': {}},
        >>         {'task': 'split_train_test', 'args': {'sample_file': 'train_samples_original.txt', 'train_ratio': 0.5}},
        >>         {'task': 'select_cls_samples', 'args': {'n': 100}},
        >>         {'task': 'pairing_intra_cls', 'args': {'src_info_file': "cls_selection_100.json", 'tgt_info_file': "test_samples.txt"}},
        >>     ]
        >> )

    """
    convert_syntra2(
        "/home/yuan/data/syntra/hismap",
        "/home/yuan/data/syntra2/hismap",
        subset="hameln",
    )

    # convert_kitti(
    #     input_path="/koko/datasets/kitti/data_scene_flow",
    #     output_path="/home/yuan/data/syntra/autonomous_driving/kitti",
    #     fg_ratio=0.05,
    # )

    # convert_cityscapes(
    #     input_path="/home/yuan/data/cityscapes",
    #     output_path="c",
    #     fg_ratio=0.05,
    # )

    # generate_for_one_dataset(
    #     converter_name="SyntraDataConverter",
    #     input_path=None,
    #     output_path="/home/yuan/data/syntra/remote_sensing/oem",
    #     fg_ratio=0.05,
    #     tasks=[
    #         {'task': 'select_cls_samples', 'args': {'n': 50}},
    #         {'task': 'pairing_intra_cls', 'args': {'src_info_file': "cls_selection_50.json", 'tgt_info_file': "cls_selection_50.json"}},
    #         {'task': 'pairing_intra_cls', 'args': {'src_info_file': "cls_selection_50.json", 'tgt_info_file': "test_samples.txt"}},
    #         {'task': 'pairing_sample_intra_cls', 'args': {
    #             'src_info_file': "train_samples.txt", 
    #             'tgt_info_file': "train_samples.txt"
    #         }},
    #         {'task': 'pairing_sample_intra_cls', 'args': {
    #             'src_info_file': "train_samples.txt", 
    #             'tgt_info_file': "test_samples.txt"
    #         }},
    #         {'task': 'pairing_sample_intra_cls', 'args': {
    #             'src_info_file': "train_samples.txt", 
    #             'tgt_info_file': "val_samples.txt"
    #         }},
    #     ]
    # )

    # from converting.forest_monitoring_trees import ForestMonitoringTreesConverter, color_map
    # convert_demo(
    #     convert_obj=ForestMonitoringTreesConverter,
    #     input_path="/home/yuan/data/forest_monitoring/tree_data",
    #     output_path="/home/yuan/data/syntra/forest_monitoring/trees_september",
    #     color_map=color_map,
    #     fg_ratio=0.05,
    #     month="september",
    #     # force_cast_img_to_uint8=True
    # )

    # from converting.acdc import ACDCConverter, color_map
    # convert_demo(
    #     convert_obj=ACDCConverter,
    #     input_path="/home/yuan/data/ACDC",
    #     output_path="/home/yuan/data/syntra_original_size/medical/acdc",
    #     color_map=color_map,
    #     fg_ratio=0.01,
    # )

    # from converting.hameln_rs import HamelnRSConverter, color_map
    # convert_demo(
    #     convert_obj=HamelnRSConverter,
    #     input_path="/home/yuan/data/labels_v3_manual_atkis_2000_2010_2022",
    #     output_path="/home/yuan/data/syntra/remote_sensing/hameln",
    #     color_map=color_map,
    #     fg_ratio=0.01,
    # )

    # hismap
    # root_dir = "/home/yuan/data/syntra/hismap"
    # for subdir in os.listdir(root_dir):
    #     convert_demo(
    #         convert_obj=SyntraDataConverter,
    #         input_path=None,
    #         output_path=f"/home/yuan/data/syntra/hismap/{subdir}",
    #         color_map=None,
    #         fg_ratio=0.05,
    #     )

    # from converting.siegfried import SiegfriedConverter, color_map
    # # convert_siegfried(
    # #     convert_obj=SiegfriedConverter,
    # #     input_path="/home/yuan/data/HisMap/Siegfried_Railway_Vineyard",
    # #     output_path="/koko/datasets/syntra/siegfried/railway",
    # #     class_name="railway",
    # #     selections=["fewshot10", "partial_1", "partial_10"],
    # #     color_map=color_map,
    # #     fg_ratio=0.01,
    # # )

    # convert_siegfried(
    #     convert_obj=SiegfriedConverter,
    #     input_path="/home/yuan/data/HisMap/Siegfried_Railway_Vineyard",
    #     output_path="/koko/datasets/syntra/siegfried/railway",
    #     class_name="railway",
    #     selections=[10, 50, 100],
    #     color_map=color_map,
    #     fg_ratio=0.001,
    # )



     


