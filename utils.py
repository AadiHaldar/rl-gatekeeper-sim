# utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_eval_results(eval_results: dict, outdir: str = None):
    names = list(eval_results.keys())
    avg_lat = [eval_results[n]['avg_latency'] for n in names]
    jains = [eval_results[n]['jain'] for n in names]
    ener = [eval_results[n]['energy'] for n in names]

    plt.figure(); plt.bar(names, avg_lat); plt.ylabel('Avg latency'); plt.title('Avg latency'); plt.xticks(rotation=45)
    if outdir: plt.savefig(f"{outdir}/eval_avg_latency.png")
    plt.show()

    plt.figure(); plt.bar(names, jains); plt.ylabel('Jain index'); plt.title('Fairness'); plt.xticks(rotation=45)
    if outdir: plt.savefig(f"{outdir}/eval_jain.png")
    plt.show()

    plt.figure(); plt.bar(names, ener); plt.ylabel('Energy (J)'); plt.title('Energy'); plt.xticks(rotation=45)
    if outdir: plt.savefig(f"{outdir}/eval_energy.png")
    plt.show()
