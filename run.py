"""Discover a causal graph with one method on one dataset and save the adjacency matrix.

Examples:
    python run.py --dataset cloud_edge_kwollect_10 --alg llm_batch --model deepseek-r1:70b --batch_size 7
    python run.py --dataset cloud_edge_kwollect_10 --alg llm_pairwise --model llama3.3:70b --relations direct
    python run.py --dataset cloud_edge_kwollect_10 --alg pcmci_plus --tau_max 1 --pc_alpha 0.01 --threshold 0.05
"""
import json
import os
import random
import re

import numpy as np
import pandas as pd

from args import LLM_ALGS, get_args
from data.init_prompts import prompts
from data.var_names_and_desc import VAR_NAMES_AND_DESC
from utils import is_dag, plot_dag

BNLEARN_DATASETS = ["asia", "child", "alarm", "insurance"]


def load_dataset(args):
    """Return (variable names, ground-truth adjacency matrix or None, data frame or None)."""
    if args.dataset in BNLEARN_DATASETS:
        import bnlearn as bn

        dag = bn.import_DAG(f"./data/{args.dataset}.bif", verbose=0)
        var_names = list(dag["adjmat"].columns)
        true_adj = dag["adjmat"].to_numpy().astype(int)
        df = bn.sampling(dag, n=args.n_samples, verbose=0)[var_names]
        return var_names, true_adj, df

    if args.dataset in VAR_NAMES_AND_DESC:
        # Monitoring metrics: no ground truth. The LLM methods only need the variable descriptions,
        # the time series (data/<dataset>.csv) is only needed for data-driven methods and --with_statistics.
        var_names = list(VAR_NAMES_AND_DESC[args.dataset].keys())
        csv_path = f"./data/{args.dataset}.csv"
        df = pd.read_csv(csv_path)[var_names] if os.path.exists(csv_path) else None
        return var_names, None, df

    raise ValueError(f"Unknown dataset {args.dataset}")


def run_llm(args, var_names, df):
    from algs.llm.batch import llm_batch
    from algs.llm.pairwise import llm_pairwise

    if df is None:
        if args.with_statistics:
            raise FileNotFoundError(f"--with_statistics needs data/{args.dataset}.csv")
        df = pd.DataFrame(columns=var_names)  # descriptions only
    common = dict(
        var_names_and_desc=VAR_NAMES_AND_DESC[args.dataset],
        prompts=prompts[args.dataset],
        df=df,
        backend=args.backend,
        model=args.model,
        relations=args.relations,
        include_statistics=args.with_statistics,
    )
    if args.alg == "llm_pairwise":
        return llm_pairwise(**common, num_ctx=args.num_ctx or 2048)
    return llm_batch(**common, batch_size=args.batch_size, num_ctx=args.num_ctx or 8192)


def run_data_driven(args, var_names, df):
    if df is None:
        raise FileNotFoundError(f"{args.alg} needs the time series in data/{args.dataset}.csv")
    data = df.to_numpy().astype(float)
    noise = np.random.normal(loc=0.0, scale=0.00001, size=data.shape)

    if args.alg == "pc":
        from algs.pc.pc import pc
        return pc(data + noise)
    if args.alg == "ges":
        from algs.ges.ges import ges
        return ges(data + noise, score_func="local_score_BIC", maxP=None, parameters=None)
    if args.alg == "fci":
        from algs.fci.fci import fci
        return fci(data + noise)
    if args.alg == "notears":
        from algs.notears.notears import notears
        return notears(data, lambda1=args.lambda1, loss_type="l2", w_threshold=args.w_threshold)
    if args.alg == "dagma_linear":
        from algs.dagma_baseline.dagma_wrapper import dagma_linear_wrapper
        return dagma_linear_wrapper(data, args.lambda1)
    if args.alg == "dagma_nonlinear":
        from algs.dagma_baseline.dagma_wrapper import dagma_nonlinear_wrapper
        return dagma_nonlinear_wrapper(data, len(var_names), args.lambda1, args.lambda2)
    if args.alg == "varlingam":
        from algs.VarLiNGAM.varlingam import varlingam_wrapper
        return varlingam_wrapper(data, lag=args.lag, threshold=args.threshold)
    if args.alg == "pcmci_plus":
        from algs.pcmci.pcmci_plus import pcmci_plus_wrapper
        return pcmci_plus_wrapper(data, var_names, tau_max=args.tau_max, pc_alpha=args.pc_alpha,
                                  p_threshold=args.threshold)
    raise ValueError(f"Unknown algorithm {args.alg}")


def run_name(args):
    if args.alg in LLM_ALGS:
        from algs.llm.clients import DEFAULT_MODELS

        model = re.sub(r"[^A-Za-z0-9.-]+", "-", args.model or DEFAULT_MODELS[args.backend])
        name = f"{args.alg}_{model}_{args.relations}"
        if args.alg == "llm_batch":
            name += f"_bs{args.batch_size}"
        return name + ("_stats" if args.with_statistics else "")
    if args.alg == "varlingam":
        return f"varlingam_lag{args.lag}_th{args.threshold}"
    if args.alg == "pcmci_plus":
        return f"pcmci_plus_tau{args.tau_max}_alpha{args.pc_alpha}_th{args.threshold}"
    if args.alg in ["notears", "dagma_linear", "dagma_nonlinear"]:
        return f"{args.alg}_lambda1={args.lambda1}"
    return args.alg


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    var_names, true_adj, df = load_dataset(args)
    if args.alg in LLM_ALGS:
        adj = run_llm(args, var_names, df)
    else:
        adj = run_data_driven(args, var_names, df)
    adj = np.asarray(adj).astype(int)

    outdir = os.path.join(args.logdir, args.dataset)
    os.makedirs(outdir, exist_ok=True)
    name = run_name(args)
    np.save(os.path.join(outdir, f"{name}.npy"), adj)

    summary = {
        "dataset": args.dataset,
        "alg": args.alg,
        "args": vars(args),
        "variables": var_names,
        "n_edges": int(adj.sum()),
        "is_dag": bool(is_dag(adj)),
    }
    if true_adj is not None:
        from metrics import compute_metrics
        summary["metrics"] = compute_metrics(true_adj, adj)
    with open(os.path.join(outdir, f"{name}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{summary['n_edges']} edges found, saved to {outdir}/{name}.npy")
    if "metrics" in summary:
        print(json.dumps(summary["metrics"], indent=2, default=str))
    if args.plot:
        plot_dag(adj, var_names, save_path=os.path.join(outdir, f"{name}.png"))


if __name__ == "__main__":
    main()
