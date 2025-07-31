from ml4co_kit import MISDataGenerator, MISSolver, draw_mis_problem

# RB
# mis_rb = MISDataGenerator(
#     num_threads=1,
#     data_type="rb",
#     train_samples_num=1,
#     val_samples_num=0,
#     test_samples_num=0,
#     save_path="problems/data/mis/rb",
#     nodes_num_min=60,
#     nodes_num_max=100,
# )
# mis_rb.generate()
# solver = MISSolver()
# solver.from_txt("problems/data/mis/rb/mis_rb-60-100.txt")
# draw_mis_problem(
#     save_path="problems/figures/mis_rb.png",
#     graph_data=solver.graph_data[0],
# )

# ba
# mis_ba = MISDataGenerator(
#     num_threads=1,
#     data_type="ba",
#     train_samples_num=1,
#     val_samples_num=0,
#     test_samples_num=0,
#     save_path="problems/data/mis/ba",
#     nodes_num_min=60,
#     nodes_num_max=100,
# )
# mis_ba.generate()
# solver = MISSolver()
# solver.from_txt("problems/data/mis/ba/mis_ba-60-100.txt")
# draw_mis_problem(
#     save_path="problems/figures/mis_ba.png",
#     graph_data=solver.graph_data[0],
# )

# hk
# mis_hk = MISDataGenerator(
#     num_threads=1,
#     data_type="hk",
#     train_samples_num=1,
#     val_samples_num=0,
#     test_samples_num=0,
#     save_path="problems/data/mis/hk",
#     nodes_num_min=60,
#     nodes_num_max=100,
# )
# mis_hk.generate()
# solver = MISSolver()
# solver.from_txt("problems/data/mis/hk/mis_hk-60-100.txt")
# draw_mis_problem(
#     save_path="problems/figures/mis_hk.png",
#     graph_data=solver.graph_data[0],
# )

# ws
# mis_ws = MISDataGenerator(
#     num_threads=1,
#     data_type="ws",
#     train_samples_num=1,
#     val_samples_num=0,
#     test_samples_num=0,
#     save_path="problems/data/mis/ws",
#     nodes_num_min=60,
#     nodes_num_max=100,
# )
# mis_ws.generate()
# solver = MISSolver()
# solver.from_txt("problems/data/mis/ws/mis_ws-60-100.txt")
# draw_mis_problem(
#     save_path="problems/figures/mis_ws.png",
#     graph_data=solver.graph_data[0],
# )

# er
mis_er = MISDataGenerator(
    num_threads=1,
    data_type="er",
    train_samples_num=1,
    val_samples_num=0,
    test_samples_num=0,
    save_path="problems/data/mis/er",
    nodes_num_min=60,
    nodes_num_max=100,
    er_prob=0.3,
)
mis_er.generate()
solver = MISSolver()
solver.from_txt("problems/data/mis/er/mis_er-60-100.txt")
draw_mis_problem(
    save_path="problems/figures/mis_er.png",
    graph_data=solver.graph_data[0],
)