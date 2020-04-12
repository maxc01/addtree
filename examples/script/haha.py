import numpy as np
import json

haha = json.load(open("./useful.json"))

iters = []
top1 = []
value = []
spars = []
for i in haha:
    iters.append(int(i["iteration"]))
    top1.append(i["obj_info"]["top1"])
    value.append(-i["obj_info"]["value"])
    spars.append(i["obj_info"]["sparsity"])

acc_value = np.maximum.accumulate(value)

fig, ax = plt.subplots()
ax.plot(iters, acc_value)
plt.show()

