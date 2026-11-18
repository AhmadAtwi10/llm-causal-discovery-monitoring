"""Batch prompting (the approach proposed in the paper).

Each variable is taken in turn as the *root*. The variables that come after it are split into batches of
`batch_size`, and for each batch the LLM is asked which of them are caused by the root. With batch size b
this needs about n^2 / (2b) queries instead of n(n-1)/2 for pairwise prompting.

relations="all"    -> direct and indirect effects (prompt shown in Appendix A of the paper)
relations="direct" -> direct effects only
"""
import re

import numpy as np
from tqdm.autonotebook import tqdm

from .clients import chat
from .utils import PreviousEdges


def _prompt_all(user, root, others):
    return f'''{user}
                We are examining the causal relationship between a root variable and a batch of other variables.

                Root variable:  {root} 
                Other variables: {others}

                Instructions:
                - Analyze whether each variable is **caused directly or indirectly** by the root variable.
                - An indirect cause means the root may influence the variable through one or more intermediate variables.
                - Only include variables that are affected by the root variable in any way (directly or indirectly).
                - Do not include the root variable in the answer.
                - The final answer must be placed within <Answer> tags and should list only the names of the variables separated by commas, with no additional text.

                Example of final answer format: <Answer>var1,var2,var3</Answer>

                Let's work this out step by step to ensure the right answer.
                '''


def _prompt_direct(user, root, others):
    return f'''{user}
                We are examining the causal relationship between the root variable and other variables.
                Here are the causal relationships you know so far:

                Which of these variables are directly caused by the root variable?

                Root variable:  {root} 
                Other variables: {others}

                Instructions:
                - You will be comparing the root variable with the other variables.
                - For each variable, check if it is directly caused by the root variable. If it is, include it in your final answer.
                - The final answer should be provided once, at the end, within the <Answer> tags. 
                - Only include the names of the variables that are directly caused by the root variable.
                - If there is no relation, leave the answer empty.
                - The root variable shoud not be in the answer.
                - **Note:** Do not compare a variable to itself and ensure that your answer only contains the variable names separated by commas (without additional text).

                Example of final answer format: <Answer>var1,var2,var3</Answer>
                Let's work this out step by step to ensure the right answer
                '''


PROMPTS = {"all": _prompt_all, "direct": _prompt_direct}


def llm_batch(var_names_and_desc, prompts, df, backend="ollama", model=None, relations="all", batch_size=7,
              include_statistics=False, num_ctx=8192):
    previous_edges = PreviousEdges()
    all_variables = list(var_names_and_desc.keys())
    visited_variables = []

    for root_variable in tqdm(all_variables):
        visited_variables.append(root_variable)
        # Each pair is asked once: a root is compared only with the variables after it in the list
        remaining_variables = [var for var in all_variables if var not in visited_variables]
        caused_variables = []

        for i in range(0, len(remaining_variables), batch_size):
            batch = remaining_variables[i:i + batch_size]

            descriptions = [f"- {root_variable}: {var_names_and_desc[root_variable].description}"]
            for var_name in batch:
                descriptions.append(f"- {var_name}: {var_names_and_desc[var_name].description}\n //")
            query = PROMPTS[relations](prompts["user"], descriptions[0], ', '.join(descriptions[1:]))

            if include_statistics:
                for var_name in batch:
                    if root_variable in df.columns and var_name in df.columns:
                        corr_coef = np.corrcoef(df[[root_variable, var_name]].to_numpy().T)[0, 1]
                        if abs(corr_coef) >= 0.1:
                            query += (f'\nThe Pearson correlation coefficient between "{root_variable}" and '
                                      f'"{var_name}" is {round(corr_coef, 2)}.\n')

            print("Query:\n", query)
            answer = chat(prompts["system"], query, backend=backend, model=model, num_ctx=num_ctx)
            print("Response:\n", answer)

            match = re.search(r"<Answer>(.*?)</Answer>", answer, re.DOTALL)
            if match:
                caused_variables.extend(match.group(1).split(","))
            else:
                print("No causal variables found in the answer.")

            caused_variables = [item.strip() for item in caused_variables if item.strip()]
            invalid = [v for v in caused_variables if v not in all_variables or v == root_variable]
            for v in invalid:
                print(f"Warning: {v} is not a valid variable name. Skipping.")
            caused_variables = [v for v in caused_variables if v not in invalid]

        for var_name in all_variables:
            choice = "A" if var_name in caused_variables else "C"  # A: root -> var, C: no edge
            previous_edges.add_edge_pair_wise(root_variable, var_name, choice)

    return previous_edges.get_adjacency_matrix(all_variables)
