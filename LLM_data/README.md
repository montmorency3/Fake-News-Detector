Data in this directory

ISOT_llm_test.csv 
- contains the version of the test partition of the ISOT dataset in the form in which it was used in the LLM prompts

batch_tasks_ISOT_gpt-4o-mini.jsonl
- contains all the queries against GPT-4o-mini based on the ISOT dataset as json objects, used for batch processing with OpenAI's API

llm_results.csv
- contains the final accuracies and invalid answers of the LLMs

res_ISOT_gpt-4o-mini.jsonl
- contains the responses to each individual article from ISOT of GPT-4o-mini as json-objects

res_ISOT_gpt-4o.csv
- contains the responses to each individual article from ISOT of GPT-4o

res_ISOT_o1-mini_max3x.csv
- contains the responses to each individual article from ISOT of o1 (with a max_completion_tokens parameter of 1024)

res_ISOT_o1-mini.csv
- contains the responses to each individual article from ISOT of o1 (with a max_completion_tokens parameter of 3 times the number of tokens in the input)

res_LIAR2_gpt-4o-mini.csv
- contains the responses to each individual statement from LIAR2 of GPT-4o-mini

res_LIAR2_gpt-4o.csv
- contains the responses to each individual statement from LIAR2 of GPT-4o

res_LIAR2_o1-mini.csv
- contains the responses to each individual statement from LIAR2 of o1  
