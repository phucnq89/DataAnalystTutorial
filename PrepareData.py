import numpy as np
import matplotlib.pyplot as plt
# dat = np.random.uniform(0.0, 5.0, 20)
# # dat = np.random.normal(5.0, 1.0, 100000)
# # Mean: 5.0, Standard deviation: 1.0
# print("Data:\n", dat)
#
# plt.hist(dat, 5)
# plt.show()

# Scatter plot
# age_of_each_car = [5,7,8,7,2,17,2,9,4,11,12,9,6]
# speed_of_each_car = [99,86,87,88,111,86,103,87,94,78,77,85,86]
age_of_each_car = np.random.normal(5.0, 1.0, 500)
speed_of_each_car = np.random.normal(85.0, 5.0, 500)
plt.scatter(age_of_each_car, speed_of_each_car)
plt.xlabel("Age of cars", fontsize=13)
plt.ylabel("Speed of cars", fontsize=13)
plt.show()