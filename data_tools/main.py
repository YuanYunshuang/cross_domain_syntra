
from converting.forest_monitoring_trees import ForestMonitoringTreesConverter

if __name__ == "__main__":
    # add python path

    input_directory = "/home/yuan/data/forest_monitoring/tree_data"
    output_directory = "/home/yuan/data/syntra/forest_monitoring/forest_monitoring.trees"


    converter = ForestMonitoringTreesConverter(
        syntra_data_path=output_directory,
        customized_data_path=input_directory,
        fg_ratio=0.05
    )
    # convert to syntra format
    converter.convert()
    # select 100 samples for train for each class
    converter.select_cls_samples(100)
    # generate intra-class paring json file with DINOv2 and faiss-indexing
    converter.paring_intra_cls("cls_selection_100.json", "cls_selection_100.json",)