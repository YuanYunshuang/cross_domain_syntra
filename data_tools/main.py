
from converting.forest_monitoring_trees import ForestMonitoringTreesConverter

if __name__ == "__main__":
    # add python path

    input_directory = "/home/yuan/data/forest_monitoring/tree_data"
    output_directory = "/home/yuan/data/syntra/forest_monitoring/forest_monitoring.trees"
    # convert(input_directory, output_directory)
    converter = ForestMonitoringTreesConverter(
        syntra_data_path=output_directory,
        fg_ratio=0.05
    )
    # converter.select_cls_samples(100)
    converter.paring_intra_cls("cls_selection_100.json", "cls_selection_100.json",)