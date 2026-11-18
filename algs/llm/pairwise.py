"""Pairwise prompting: one LLM query per pair of variables (A -> B, B -> A or no relation).

For n variables this needs n(n-1)/2 queries. The prompt is the one shown in Appendix A of the paper.
"""
import random
from itertools import combinations

import numpy as np
from tqdm.autonotebook import tqdm

from .clients import chat
from .utils import PreviousEdges

# relations="direct": only direct edges; relations="all": direct and indirect (transitive) edges
RELATION_NOTE = {
    "direct": "Direct relation only.",
    "all": "Consider both direct and indirect relations.",
}
NO_RELATION = {
    "direct": "direct",
    "all": "direct or indirect",
}


def llm_pairwise(var_names_and_desc, prompts, df, backend="ollama", model=None, relations="direct",
                 include_statistics=False, num_ctx=2048):
    # List all edges
    edges = list(combinations(df.columns, 2))
    previous_edges = PreviousEdges()
    for e in tqdm(edges):
        if random.random() < 0.5:
            head, tail = e
        else:
            tail, head = e

        head_name = head
        tail_name = tail        
        head = var_names_and_desc[head]
        tail = var_names_and_desc[tail]   

        query = f'''{prompts["user"]} 
        We are examining the causal relationship between two variables:

        - {head.name}: {head.description}
        - {tail.name}: {tail.description}
                
        Here are the causal relationships you know so far:
        {previous_edges.get_previous_relevant_edges_string(head.name, tail.name)}
        '''
        if include_statistics and head_name in df.columns and tail_name in df.columns:
            arr = df[[head_name, tail_name]].to_numpy().T
            corr_coef = np.corrcoef(arr)[0, 1]
            if abs(corr_coef) < 0.1:
                print(f"Skipping pair {head_name} and {tail_name} due to low: ({corr_coef})")
                continue
            else:
                corr_coef = round(corr_coef, 2)
                query += f'''
                To help you, the Pearson correlation coefficient between "{head_name}" and "{tail_name}" is {corr_coef}.
                '''

        query += f'''
        Which cause-and-effect relationship is more likely? Note: {RELATION_NOTE[relations]}
        A. "{head.name}" causes "{tail.name}". 
        B. "{tail.name}" causes "{head.name}". 
        C. There is no {NO_RELATION[relations]} causal relationship between "{head.name}" and "{tail.name}".
        Let's work this out step by step to ensure the right answer. 
        Then provide your final answer within the tags <Answer>A/B/C</Answer>.'''
        # Query formatting
        print("Query:\n", query)

        answer = chat(prompts["system"], query, backend=backend, model=model, num_ctx=num_ctx)
        print("Response:\n", answer)

        idx = answer.find('<Answer>')
        if idx == -1:
            print("NO ANSWER FOUND")
            print("This was the answer:", answer)
            continue
        choice = answer[idx + 8]

        previous_edges.add_edge_pair_wise(head_name, tail_name, choice)

    keys = list(var_names_and_desc.keys())
    adj_matrix = previous_edges.get_adjacency_matrix(keys)
    return adj_matrix