import matplotlib.pyplot as plt

x = [10, 20, 30, 40, 50, 60]
y = [13, 45, 23, 34, 96, 76]

plt.title("Bar Graph")
plt.bar(x, y, color='red', width=5)
plt.xlabel("X values")
plt.ylabel("Y values")
plt.show()