# Variance
n_num = [1, 2, 3, 4, 5]

n = len(n_num)
mean = sum(n_num) / n

variance = sum((x - mean) ** 2 for x in n_num) / n

print("Variance is :", variance)