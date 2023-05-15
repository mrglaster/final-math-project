import numpy as np
from pulp import *

amount_of_weeks = 4

demand_matrix = [[5, 3, 2],
                 [7, 4, 3],
                 [8, 6, 4],
                 [10, 7, 5]]

producing_prices = [[21, 15, 20],
                    [12, 18, 25],
                    [15, 12, 30],
                    [20, 25, 35]]

storage_price = [[2, 3, 4],
                 [3, 4, 5],
                 [4, 5, 6],
                 [5, 6, 7]]

testing_times = [2, 3, 5]
product_volumes = [4, 6, 10]
storage_volume = 500
max_assemblies = 50
max_test_time = 75


def compute_computer_assembly(amount_of_weeks: int, demand_matrix: [], producing_prices: [],
                              storage_price: [], testing_times: [], product_volumes: [],
                              storage_volume: int, max_assemblies: int, max_test_time: int, transpose=False) -> tuple:
    """Finds the best solution """
    # Initializing Problem
    model = LpProblem("Computers_Assembly", LpMinimize)

    # Initializing solution array
    x = [[LpVariable(f"x_{i}_{j}", lowBound=0, cat="Integer") for j in range(3)] for i in range(amount_of_weeks)]


    # Initializing I matrix
    eow_left = [[LpVariable(f"I_{i}_{j}", lowBound=0, cat="Integer") for j in range(3)] for i in range(amount_of_weeks)]
    for i in range(1, amount_of_weeks):
        for j in range(3):
            eow_left[i][j] = LpVariable(f"I_{i}_{j}", lowBound=0, cat="Integer")
            if demand_matrix[i][j] == 0:
                model += eow_left[i][j] == 0
            else:
                # Otherwise, update eow_left[i][j] based on the formula you provided
                model += eow_left[i][j] == x[i][j] - demand_matrix[i][j] + eow_left[i-1][j]



    # Setting up the minimization function
    model += lpSum(
        producing_prices[i][j] * x[i][j] + storage_price[i][j] * eow_left[i][j] for i in range(amount_of_weeks) for j in
        range(3))

    # Setting up the first restriction
    for i in range(amount_of_weeks):
        for j in range(3):
            model += x[i][j] + eow_left[i - 1][j] >= demand_matrix[i][j] + eow_left[i][j]


    # Setting up the max assemblies limit
    for i in range(amount_of_weeks):
        model += lpSum(x[i][j] for j in range(3)) <= max_assemblies

    # Setting up the test time limits
    for i in range(amount_of_weeks):
        model += lpSum(testing_times[j] * x[i][j] for j in range(3)) <= max_test_time

    # Setting up the storage volume restriction
    for i in range(amount_of_weeks):
        model += lpSum(product_volumes[j] * x[i][j] for j in range(3)) <= storage_volume

    result = model.solve(PULP_CBC_CMD(msg=False))

    if result != LpStatusInfeasible:
        result_array = []
        for i in range(amount_of_weeks):
            current_line = []
            for j in range(3):
                current_line.append(int(value(x[i][j])))
            result_array.append(current_line)
        if transpose:
            result_array = np.transpose(result_array)
        return value(model.objective), result_array, "The solution was successfully found!"
    return None, None, "Unable to find the solution!"


if __name__ == '__main__':
    a, b, c = compute_computer_assembly(amount_of_weeks=amount_of_weeks, demand_matrix=demand_matrix,
                                        producing_prices=producing_prices, storage_price=storage_price,
                                        testing_times=testing_times, product_volumes=product_volumes,
                                        storage_volume=storage_volume, max_assemblies=max_assemblies,
                                        max_test_time=max_test_time, transpose=True)
    print(f"{a}     :    {b}        :  {c}")

