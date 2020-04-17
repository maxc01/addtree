import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from tqdm import tqdm
import seaborn as sns


def get_algo_expid(name):
    base_name = os.path.basename(name)
    algo, expid = os.path.splitext(base_name)[0].split("-")
    return algo, expid


all_results = []
results_path = "./compare_output_new"
names = glob.glob(os.path.join(results_path, "*.json"))
for name in names:
    algo, expid = get_algo_expid(name)
    with open(name) as f:
        info = json.load(f)

    df_exp = pd.DataFrame(info)
    # inject extra information for this experiment
    df_exp["algo"] = algo
    df_exp["expid"] = expid

    all_results.append(df_exp)

df = pd.concat(all_results, axis=0)

### preprocess df
# 1. sort by iterations in each group
df_EQ300 = df.sort_values(["algo", "expid", "iteration"]).groupby(["algo", "expid"]).filter(lambda x: len(x)==300)
df_EQ300["cummax"] = df_EQ300.groupby(['algo', 'expid'])["value"].cummax()

fig, ax = plt.subplots()
sns.lineplot(x="iteration", y="cummax", hue="algo", data=df_EQ300, ci=95, ax=ax)
plt.show()


# df.groupby(["algo","expid"]).agg({"value": "count"}).rename(columns={'value':"Count"})

### data exploration


def add_plot(df, algo, ax):
    df_algo = df.query(f'algo=="{algo}"').copy()
    df_algo["cummax"] = df_algo.groupby("expid")["value"].cummax()
    sns.lineplot(
        x="iteration", y="cummax", ci=None, data=df_algo, ax=ax, label=f"{algo}"
    )


fig, ax = plt.subplots()
for algo in ["tpe", "random", "addtree"]:
    add_plot(df, algo, ax)

plt.show()


###
first_n = 300
over_200 = df.groupby(["algo", "expid"]).filter(lambda x: len(x) >= first_n)
over_200_first_200 = over_200[over_200["iteration"] <= first_n]
# over_200.sort_values(['algo', 'expid', 'iteration']).groupby(["algo", "expid"])

###
fig, ax = plt.subplots()
sns.lineplot(x="iteration", y="value", hue="algo", data=over_200_first_200, ax=ax)
plt.show()


### eaiser way to select the right groups
G10 = over_200.sample(frac=1).groupby("algo").head(300 * 10)
fig, ax = plt.subplots()
sns.lineplot(x="iteration", y="value", hue="algo", data=G10, ci=95, ax=ax)
plt.show()

fig, ax = plt.subplots()
for algo in ["tpe", "random", "addtree"]:
    add_plot(G10, algo, ax)
plt.show()
