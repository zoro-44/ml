import statistics

data = [10, 20, 30, 40, 50]

mean = statistics.mean(data)
median = statistics.median(data)
variance = statistics.variance(data)
stdev = statistics.stdev(data)

print("Mean:", mean)
print("Median:", median)
print("Variance:", variance)
print("Standard Deviation:", stdev)