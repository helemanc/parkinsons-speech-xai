import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from scipy.stats import pearsonr

plt.rc("text", usetex=False)

df = pd.read_csv("parsed_results_hubert_overlap_.csv")
df = df.drop(
    columns=[
        "sensitivity_mean",
        "sensitivity_std",
        "specificity_mean",
        "specificity_std",
        "Unnamed: 0",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
    ]
)

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

parsed = {}
for index, row in df.iterrows():
    template = ""
    template += f"{methods[row['m1'].split('/')[-1]]} / {methods[row['m2']]}"
    # template += ' & '

    for k in columns:
        template += " & "
        if k != "faithfulness":
            template += f"{row[f'{k}_mean']:.2f} $\\pm$ {row[f'{k}_std']:.2f}"
        else:
            template += f"{row[f'{k}_mean']:.3f} $\\pm$ {row[f'{k}_std']:.3f}"

    template += " \\\\"
    parsed[f"{methods[row['m1'].split('/')[-1]]} / {methods[row['m2']]}"] = template
    print(template)


plt.style.use(["ieee", "science"])

plt.subplot(121)
# plt.suptitle("Correlation between IoU and faithfulness metrics")

p = pearsonr(df["iou_mean"], df["faithfulness_mean"])
title1 = rf"$\rho$={p.statistic:.2f}, p-value $<$ 0.001"
plt.scatter(df["iou_mean"], 1000 * df["faithfulness_mean"])
plt.title(title1)
plt.ylabel(r"FF [$10^{-3}$]")
plt.xlabel("IoU")

p = pearsonr(df["iou_mean"], df["AD_mean"])
print(p)
title2 = rf"$\rho$={p.statistic:.2f}, p-value $<$ 0.001"
plt.subplot(122)
plt.title(title2)
plt.scatter(df["iou_mean"], df["AD_mean"])
plt.ylabel("AD")
plt.xlabel("IoU")

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig("scatter.pdf")
