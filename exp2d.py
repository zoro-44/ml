from scipy.linalg import solve

A = [[2, 1], [1, 3]]
b = [5, 6]

solution = solve(A, b)

print("Solution:", solution)
print("x =", solution[0])
print("y =", solution[1])