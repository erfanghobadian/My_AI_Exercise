import numpy as np

"""
How to Calculate dm and db
dm is Partial Derivative of Cost Function to m  
db is Partial Derivative of Cost Function to b

Cost Function = 1/n * Sigma( (y - y_predicted) **2 )  

dm = -2/n * sigma(x * (y - y_predicted) )
db = -2/n * sigma( (y - y_predicted) )

"""


def gradient(x, y) -> None:
    current_m, current_b = 0, 0
    iteration = 10000
    learning_rate = 0.046
    n = len(x)
    for i in range(iteration):
        y_predicted = current_m * x + current_b
        current_cost = 1 / n * sum([val ** 2 for val in (y - y_predicted)])
        dm = -2 / n * sum(x * (y - y_predicted))
        db = -2 / n * sum((y - y_predicted))
        current_m -= learning_rate * dm
        current_b -= learning_rate * db
        print(current_m, current_b, current_cost)


x = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([5, 7, 9, 11, 13, 15, 17])
gradient(x, y)
