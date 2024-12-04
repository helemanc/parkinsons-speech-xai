import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from scipy.stats import pearsonr

plt.rc("text", usetex=False)

df = pd.read_csv("sel_ff.csv")
columns = [
    "AI",
    "AD",
    "AG",
    "faithfulness",
    "inp_fid",
    "sparseness",
    "complexity",
    "iou",
]

methods = {
    "ggc": "Guided GradCAM",
    "gbp": "Guided Backprop",
    "ig": "Integrated Gradients",
    "shap": "Gradient SHAP",
    "smoothgrad": "Smoothgrad",
    "saliency": "Saliency",
}

plt.style.use(["ieee", "science"])

plt.subplot(121)
plt.suptitle("Correlation between IoU and faithfulness metrics")

p = pearsonr(df["FF"], df["sacc"])
print(p)
title1 = rf"$\rho$={p.statistic:.2f}, p-value$=${p.pvalue:.3f}"
plt.scatter(df["FF"], df["sacc"])
plt.title(title1)
plt.ylabel(r"Selective Accuracy")
plt.xlabel("FF $[10^{-3}]$")

p = pearsonr(df["FF"], df["sf1"])
print(p)
title1 = rf"$\rho$={p.statistic:.2f}, p-value$=${p.pvalue:.3f}"
plt.subplot(122)
plt.scatter(1000 * df["FF"], df["sf1"])
plt.title(title1)
plt.ylabel(r"Selective F1-score")
plt.xlabel("FF $[10^{-3}]$")

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig("sel_ff.pdf")
