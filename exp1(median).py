# Median
n_num = [6, 7, 8, 9, 10]

n_num.sort()
n = len(n_num)

if n % 2 == 0:
    median1 = n_num[n // 2]
    median2 = n_num[n // 2 - 1]
    median = (median1 + median2) / 2
else:
    median = n_num[n // 2]

print("Median is :", median)