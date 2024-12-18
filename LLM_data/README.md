# Data in This Directory

This directory contains several files related to LLM-based processing of the ISOT and LIAR2 datasets. Below is a detailed description of each file:

- **llm_results.csv**
  - Contains the final accuracies and invalid answers of the LLMs.

## ISOT Dataset Files

- **ISOT_llm_test.csv**
  - Contains the version of the test split of the ISOT dataset in the form in which it was used in the LLM prompts.

- **batch_tasks_ISOT_gpt-4o-mini.jsonl**
  - Contains all the queries against GPT-4o-mini based on the ISOT dataset as JSON objects, used for batch processing with OpenAI's API.

- **res_ISOT_gpt-4o-mini.jsonl**
  - Contains the responses to each individual article from ISOT by GPT-4o-mini as JSON objects.

- **res_ISOT_gpt-4o.csv**
  - Contains the responses to each individual article from ISOT by GPT-4o.

- **res_ISOT_o1-mini.csv**
  - Contains the responses to each individual article from ISOT by o1 with a `max_completion_tokens` parameter of 1024.

- **res_ISOT_o1-mini_max3x.csv**
  - Contains the responses to each individual article from ISOT by o1 with a `max_completion_tokens` parameter of 3 times the number of tokens in the input.

## LIAR2 Dataset Files

- **res_LIAR2_gpt-4o-mini.csv**
  - Contains the responses to each individual statement from LIAR2 by GPT-4o-mini.

- **res_LIAR2_gpt-4o.csv**
  - Contains the responses to each individual statement from LIAR2 by GPT-4o.

- **res_LIAR2_o1-mini.csv**
  - Contains the responses to each individual statement from LIAR2 by o1.

---
