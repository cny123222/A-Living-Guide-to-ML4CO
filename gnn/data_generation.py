from ml4co_kit import TSPDataGenerator

# initialization
tsp_data_concorde = TSPDataGenerator(
    num_threads=8,
    nodes_num=20,
    data_type="gaussian",
    solver="LKH",
    train_samples_num=1280,
    val_samples_num=128,
    test_samples_num=128,
    save_path="gnn/data"
)

# generate
tsp_data_concorde.generate()