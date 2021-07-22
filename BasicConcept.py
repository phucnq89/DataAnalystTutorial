import numpy as np
from scipy import stats

# Mean, Median, Mode
speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
x = np.mean(speed)
print("Mean: ", x)
print("Median: ", np.median(speed))
m = stats.mode(speed)
print("Mode: ", m)

# Standard Deviation, Variance
speed2 = [86, 87, 88, 86, 87, 85, 86]
std = np.std(speed2)
print("Standard deviation: ", std)
print("Variance: ", np.var(speed2))

# Percentile
ages = [5, 31, 43, 48, 50, 41, 7, 11, 15, 39, 80, 82, 32, 2]
p = np.percentile(ages, 90)
print(p) # 90% ds từ 71 tuổi trở xuống