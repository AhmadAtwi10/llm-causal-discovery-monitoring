import argparse

LLM_ALGS = ["llm_pairwise", "llm_batch"]
DATA_ALGS = ["pc", "ges", "fci", "notears", "dagma_linear", "dagma_nonlinear", "varlingam", "pcmci_plus"]


def get_args():
    parser = argparse.ArgumentParser(description="LLM-driven causal discovery for monitoring metrics")
    parser.add_argument("--dataset", type=str, default="cloud_edge_kwollect_10",
                        help="cloud_edge_kwollect_10 | cloud_edge_kwollect_30 | cloud_edge_kwollect_100 | "
                             "asia | child | alarm | insurance")
    parser.add_argument("--alg", type=str, default="llm_batch", choices=LLM_ALGS + DATA_ALGS,
                        help="causal discovery method")
    parser.add_argument("--logdir", type=str, default="./logs", help="where results are written")
    parser.add_argument("--seed", type=int, default=0, help="random seed (pair order, data sampling)")
    parser.add_argument("--plot", action="store_true", help="save a picture of the discovered graph")

    llm = parser.add_argument_group("LLM methods")
    llm.add_argument("--backend", type=str, default="ollama", choices=["ollama", "openai"])
    llm.add_argument("--model", type=str, default=None,
                     help="model name, e.g. llama3.3:70b, gemma3:27b, deepseek-r1:70b, mistral, mixtral:8x7b "
                          "(ollama) or gpt-4-1106-preview (openai)")
    llm.add_argument("--relations", type=str, default="all", choices=["direct", "all"],
                     help="'direct': direct edges only, 'all': direct and indirect edges")
    llm.add_argument("--batch_size", type=int, default=7, help="variables per query for llm_batch")
    llm.add_argument("--num_ctx", type=int, default=None,
                     help="Ollama context length (default 2048 for pairwise, 8192 for batch)")
    llm.add_argument("--with_statistics", action="store_true",
                     help="add Pearson correlations to the prompts (needs the data CSV)")

    data = parser.add_argument_group("data-driven methods")
    data.add_argument("--n_samples", type=int, default=1000, help="samples drawn for the bnlearn benchmarks")
    data.add_argument("--lambda1", type=float, default=0.01, help="lambda for NOTEARS and DAGMA")
    data.add_argument("--lambda2", type=float, default=0.005, help="lambda2 for DAGMA")
    data.add_argument("--w_threshold", type=float, default=0.3, help="weight threshold for NOTEARS")
    data.add_argument("--lag", type=int, default=1, help="number of lags for VarLiNGAM")
    data.add_argument("--tau_max", type=int, default=3, help="maximum lag for PCMCI+")
    data.add_argument("--pc_alpha", type=float, default=0.05, help="PCMCI+ conditional independence level")
    data.add_argument("--threshold", type=float, default=0.05,
                      help="binarization threshold: p-value for PCMCI+, coefficient for VarLiNGAM")
    return parser.parse_args()
