# Mode
from collections import Counter

n_num = [1, 2, 3, 4, 5, 5]

data = Counter(n_num)
get_mode = dict(data)

mode = [k for k, v in get_mode.items() if v == max(data.values())]

if len(mode) == len(n_num):
    print("No mode found")
else:
    print("Mode is :", mode)