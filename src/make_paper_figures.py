from pathlib import Path
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def architecture(output: Path):
    fig, ax = plt.subplots(figsize=(10.5, 3.0)); ax.set_xlim(0, 10.5); ax.set_ylim(0, 3); ax.axis("off")
    boxes = [
        (0.2, 0.45, 2.0, 2.0, "IoT clients\nShared extractor + task heads\nPer-client/task residuals"),
        (2.8, 0.45, 2.0, 2.0, "Local multi-task update\nCancellation-safe probe\nShared-layer freeze decision"),
        (5.4, 0.45, 2.0, 2.0, "Task-aware TopK\nSimilarity overlap removal\nIndex-aware sparse payload"),
        (8.0, 0.45, 2.2, 2.0, "Budgeted scheduler + server\n24-client cap, floor 8\nWeighted aggregation"),
    ]
    colors = ["#e3f2fd", "#e8f5e9", "#fff3e0", "#f3e5f5"]
    for (x,y,w,h,label), color in zip(boxes, colors):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04",fc=color,ec="#37474f",lw=1.2))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=9)
    for i in range(3):
        ax.add_patch(FancyArrowPatch((boxes[i][0]+boxes[i][2],1.45),(boxes[i+1][0],1.45),arrowstyle="-|>",mutation_scale=13,lw=1.2))
    ax.add_patch(FancyArrowPatch((9.1,0.42),(1.2,0.42),connectionstyle="arc3,rad=-0.18",arrowstyle="-|>",mutation_scale=13,lw=1.1))
    ax.text(5.2,0.02,"broadcast updated shared extractor and task heads",ha="center",fontsize=8)
    fig.tight_layout(); fig.savefig(output, bbox_inches="tight"); plt.close(fig)


def convergence(source: Path, output: Path):
    rows=list(csv.DictReader(source.open())); grouped=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r["method"] not in {"fedavg","topksgd","scaffold","proposed"}: continue
        grouped[(r["dataset"],r["method"])][int(r["round"])].append(float(r["task1_metric"]))
    fig,axes=plt.subplots(1,3,figsize=(11,3.2))
    names={"fedavg":"FedAvg","topksgd":"TopK-SGD","scaffold":"SCAFFOLD","proposed":"Proposed"}
    colors={"fedavg":"#546e7a","topksgd":"#ef6c00","scaffold":"#1565c0","proposed":"#c62828"}
    for ax,dataset in zip(axes,["uci_har","cwru","intel_sensor"]):
        for method in names:
            series=grouped[(dataset,method)]; rounds=sorted(series)
            mean=np.array([np.mean(series[r]) for r in rounds]); sd=np.array([np.std(series[r],ddof=1) for r in rounds])
            ax.plot(rounds,mean,label=names[method],color=colors[method],lw=1.3)
            ax.fill_between(rounds,mean-sd,mean+sd,color=colors[method],alpha=.10)
        ax.set_title(dataset.replace('_',' ').upper()); ax.set_xlabel("Round")
        ax.set_ylabel("MSE" if dataset=="intel_sensor" else "Accuracy")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(output,bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    root=Path(__file__).resolve().parents[1]; paper=root/"paper"; paper.mkdir(exist_ok=True)
    architecture(paper/"fig_architecture.pdf")
    convergence(root/"results/analysis/all_curves.csv",paper/"fig_convergence.pdf")
