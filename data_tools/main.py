
from converting.forest_monitoring_trees import ForestMonitoringTreesConverter


def generate_for_one_dataset(input_path, output_path, tasks):
    converter = ForestMonitoringTreesConverter(
        syntra_data_path=output_path,
        customized_data_path=input_path,
        fg_ratio=0.05
    )

    for t, args in tasks.items():
        assert t in ['convert', 'select_cls_samples', 'pairing_intra_cls'], f"Unknown task: {t}"
        getattr(converter, t)(**args)


if __name__ == "__main__":
    # add python path
    input_directory = None # "/home/yuan/data/forest_monitoring/tree_data"
    # output_directory = "/home/yuan/data/syntra/geology/seismic_facies.f3"
    output_directory = "/home/yuan/data/syntra/forest_monitoring/forest_monitoring.trees"


    generate_for_one_dataset(
        input_path=input_directory,
        output_path=output_directory,
        tasks={

            # 'convert': {}, # convert to syntra format
            # 'select_cls_samples': {'n': 100}, # select 100 samples for train for each class
            'pairing_intra_cls': {  # generate intra-class paring json file with DINOv2 and faiss-indexing
                'src_info_file': "train_samples.txt",
                'tgt_info_file': "test_samples.txt",
            },
            # 'pairing_intra_cls': {
            #     'src_info_file': "cls_selection_100.json",
            #     'tgt_info_file': "test_samples.txt",
            # },
        }
    )
