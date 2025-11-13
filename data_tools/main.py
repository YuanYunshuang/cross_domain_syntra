
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
        assert t['task'] in ['convert', 'select_cls_samples', 'pairing_intra_cls', 'split_train_test'], f"Unknown task: {t}"
        getattr(converter, t['task'])(**t['args'])


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
        >>     tasks={
        >>         {'task': 'convert', 'args': {}},
        >>         {'task': 'split_train_test', 'args': {'sample_file': 'train_samples_original.txt', 'train_ratio': 0.5}},
        >>         {'task': 'select_cls_samples', 'args': {'n': 100}},
        >>         {'task': 'pairing_intra_cls', 'args': {'src_info_file': "cls_selection_100.json", 'tgt_info_file': "test_samples.txt"}},
        >>     }
        >> )

    """

    convert_kitti(
        input_path="/koko/datasets/kitti/data_scene_flow",
        output_path="/home/yuan/data/syntra/autonomous_driving/kitti",
        fg_ratio=0.05,
    )

    convert_cityscapes(
        input_path="/home/yuan/data/cityscapes",
        output_path="/home/yuan/data/syntra/autonomous_driving/cityscapes",
        fg_ratio=0.05,
    )
