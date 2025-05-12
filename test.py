import numpy as np
value_list = [0.01, 0.01, 0.09]

sum_value = np.cumsum([0] + value_list)
ap = np.linspace(0, sum_value[-1], len(value_list) + 1)[1:-1]
print(sum_value, ap, ap.size)