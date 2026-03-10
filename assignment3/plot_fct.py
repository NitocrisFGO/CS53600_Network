import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files = {
    "A2Linear": "a3_a2linear.csv",
    "Cubic": "a3_cubic.csv",
    "NewReno": "a3_newreno.csv"
}

avg = []
p99 = []

for name, file in files.items():
    df = pd.read_csv(file)

    fct = df["fct_sec"]

    avg.append(fct.mean())
    p99.append(np.percentile(fct, 99))

labels = list(files.keys())

plt.figure()
plt.bar(labels, avg)
plt.ylabel("Average FCT (s)")
plt.title("Average Flow Completion Time")
plt.savefig("avg_fct.png")

plt.figure()
plt.bar(labels, p99)
plt.ylabel("P99 FCT (s)")
plt.title("99th Percentile Flow Completion Time")
plt.savefig("p99_fct.png")

print("avg:", avg)
print("p99:", p99)