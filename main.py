import data as dt
import pso as ps
import neural_network as nn


# Data preprocessing
dt.first_data_handle()
print("data 1")
dt.products_with_sales()
print("data 2")
dt.create_week_data()
print("data 3")
dt.create_nn_data()
print("data 4")
dt.pso_data()
print("data 5")


# Test the neural network
nn.nn_testing()


# Optimize prices using particle swarm optimization
ps.optimize_prices()
