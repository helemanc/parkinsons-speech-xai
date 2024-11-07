import glob
import json
import sys

import pandas as pd

rfolder = str(sys.argv[1])

if __name__ == "__main__":
    print("+++ Parsing folder: ", rfolder, "+++")

    df = []
    for pair in glob.glob(f"{rfolder}/*"):
        try:
            m1, m2 = pair.split("_")
        except:
            m1 = m2 = pair

        try:
            man = {}
            with open(f"{pair}/average_metrics.json", "r") as f:
                metrics = json.load(f)

            for k in metrics:
                man[f"{k}_mean"] = metrics[k]["mean"]
                man[f"{k}_std"] = metrics[k]["std"]
            man.update({"m1": m1, "m2": m2})

            df.append(man)
        except:
            print(f"Ignoring {pair}. File not found.")

    df = pd.DataFrame.from_records(df)
    breakpoint()
    print(df.head())

    rfolder = rfolder.replace("/", "_")
    df.to_csv(f"parsed_{rfolder}.csv")
