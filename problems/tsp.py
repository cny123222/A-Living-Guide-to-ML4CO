from ml4co_kit import TSPDataGenerator, TSPSolver, draw_tsp_problem

# uniform
# tsp_uniform = TSPDataGenerator(
#     num_threads=1,
#     nodes_num=500,
#     data_type="uniform",
#     train_samples_num=1,
#     val_samples_num=0,
#     test_samples_num=0,
#     save_path="problems/data/tsp/uniform"
# )
# tsp_uniform.generate()
# solver = TSPSolver()
# solver.from_txt("problems/data/tsp/uniform/tsp500_uniform.txt")
# draw_tsp_problem(
#     save_path="problems/figures/tsp_uniform.png",
#     points=solver.points,
#     node_size=25,
# )

# gaussian
# tsp_gaussian = TSPDataGenerator(
#     num_threads=1,
#     nodes_num=500,
#     data_type="gaussian",
#     train_samples_num=1,
#     val_samples_num=0,
#     test_samples_num=0,
#     save_path="problems/data/tsp/gaussian"
# )
# tsp_gaussian.generate()
# solver = TSPSolver()
# solver.from_txt("problems/data/tsp/gaussian/tsp500_gaussian.txt")
# draw_tsp_problem(
#     save_path="problems/figures/tsp_gaussian.png",
#     points=solver.points,
#     node_size=25,
# )

# cluster
tsp_cluster = TSPDataGenerator(
    num_threads=1,
    nodes_num=500,
    data_type="cluster",
    train_samples_num=1,
    val_samples_num=0,
    test_samples_num=0,
    save_path="problems/data/tsp/cluster"
)
tsp_cluster.generate()
solver = TSPSolver()
solver.from_txt("problems/data/tsp/cluster/tsp500_cluster.txt")
draw_tsp_problem(
    save_path="problems/figures/tsp_cluster.png",
    points=solver.points,
    node_size=25,
)