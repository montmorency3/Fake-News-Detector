import json
import pandas as pd
import numpy as np

########################################################################################################################
# This file analyses the outputs from the OpenAI models
# - results and other useful information are printed
########################################################################################################################


########################################################################################################################
############################################# Analysis for ISOT dataset ################################################
########################################################################################################################

# load original dataset
df = pd.read_csv('../LLM_data/ISOT_llm_test.csv')

######################################## 4o-mini RESULTS ###############################################################
# Loading data from saved file
result_file_name = '../LLM_data/res_ISOT_gpt-4o-mini.jsonl'
results = []
with open(result_file_name, 'r') as file:
    for line in file:
        # Parsing the JSON string into a dict and appending to the list of results
        json_object = json.loads(line.strip())
        results.append(json_object)

# Reading the results
four_o_mini_num_correct = 0 # Instances answered correctly
four_o_mini_num_invalid = 0 # Instances with answers other than 0 or 1
total_prompt_tokens = 0
for res in results:
    task_id = res['custom_id']
    idx = task_id.split('-')[-1]
    label = task_id.split('-')[-2]
    result = res['response']['body']['choices'][0]['message']['content']
    prompt_tokens = res['response']['body']['usage']['prompt_tokens']
    total_prompt_tokens += prompt_tokens
    if result == "0" or result == "1":
        if result == label:
            four_o_mini_num_correct += 1
    else:
        four_o_mini_num_invalid += 1

print("MODEL: GPT-4o-mini")
print("Dataset: ISOT")
print(f"Accuracy: {four_o_mini_num_correct/len(results)}\nInvalid Answers: {four_o_mini_num_invalid}")
print(f"Cleaned Accuracy: {four_o_mini_num_correct/(len(results) - four_o_mini_num_invalid)}")
print(f"Total Input Tokens: {total_prompt_tokens}")


############################################# 4o RESULTS ###############################################################

gpt_four_o_file_name = '../LLM_data/res_ISOT_gpt-4o.csv'
four_o_data = pd.read_csv(gpt_four_o_file_name)
four_o_num_correct = 0
four_o_num_invalid = 0
#four_o_num_valid = 0
for idx, row in four_o_data.iterrows():
    result = row['prediction']
    label = row['label']
    if result == "0" or result == "1":
        #four_o_num_valid += 1
        if int(result) == label:
            four_o_num_correct += 1
    else:
        four_o_num_invalid += 1

print("\nMODEL: GPT-4o")
print("Dataset: ISOT")
print(f"Accuracy: {four_o_num_correct/len(four_o_data)}\nInvalid Answers: {four_o_num_invalid/len(four_o_data)}")
print(f"Cleaned Accuracy: {four_o_num_correct/(len(four_o_data) - four_o_num_invalid)}")


############################################# o1 RESULTS ###############################################################
# uses 1024 as max_completion_tokens

o_one_mini_file_name = '../LLM_data/res_ISOT_o1-mini.csv'
o_one_mini_data = pd.read_csv(o_one_mini_file_name)
o_one_mini_num_correct = 0
o_one_mini_num_invalid = 0
#o_one_num_valid = 0

# For Analysis of instances for which 1024 was beneficial (<=341 tokens) vs detrimental (>341 tokens)
o_one_mini_data['prompt_tokens'] = df['prompt_tokens']
num_benefit = 0
num_impair = 0
num_benefit_correct = 0
num_impair_correct = 0

for idx, row in o_one_mini_data.iterrows():
    result = row['prediction']
    label = row['label']
    prompt_tokens = row['prompt_tokens']
    if result == "0" or result == "1":
        #o_one_num_valid += 1
        if int(result) == label:
            o_one_mini_num_correct += 1
            # Count for both groups separately
            if 3 * prompt_tokens <= 1024:
                num_benefit_correct += 1
            else:
                num_impair_correct += 1
    else:
        o_one_mini_num_invalid += 1
    if 3*prompt_tokens <= 1024:
        num_benefit += 1
    else:
        num_impair += 1

print("\nMODEL: o1-mini_max1024")
print("Dataset: ISOT")
print(f"Accuracy: {o_one_mini_num_correct/len(o_one_mini_data)}\n"
      f"Invalid Answers: {o_one_mini_num_invalid/len(o_one_mini_data)}")
print(f"Cleaned Accuracy: {o_one_mini_num_correct/(len(o_one_mini_data) - o_one_mini_num_invalid)}")
print(f"Accuracy (<=341 tokens, benefit): {num_benefit_correct/num_benefit} ({num_benefit} instances)\n"
      f"Accuracy (>341 tokens, impaired): {num_impair_correct/num_impair} ({num_impair} instances)")


############################################# o1 RESULTS ###############################################################
# uses 3*prompt_tokens as max_completion_tokens

o_one_mini_max3x_file_name = '../LLM_data/res_ISOT_o1-mini_max3x.csv'
o_one_mini_max3x_data = pd.read_csv(o_one_mini_max3x_file_name)

o_one_mini_max3x_num_correct = 0
o_one_mini_max3x_num_invalid = 0
#o_one_num_valid = 0

# For Analysis of instances for which 3x was beneficial (>341 tokens) vs detrimental (<=341 tokens)
o_one_mini_max3x_data['prompt_tokens'] = df['prompt_tokens']
num_benefit = 0
num_impair = 0
num_benefit_correct = 0
num_impair_correct = 0

for idx, row in o_one_mini_max3x_data.iterrows():
    result = row['prediction']
    label = row['label']
    prompt_tokens = row['prompt_tokens']
    if result == "0" or result == "1":
        #o_one_num_valid += 1
        if int(result) == label:
            o_one_mini_max3x_num_correct += 1
            # Count for both groups separately
            if 3*prompt_tokens > 1024:
                num_benefit_correct += 1
            else:
                num_impair_correct += 1
    else:
        o_one_mini_max3x_num_invalid += 1
    if 3*prompt_tokens > 1024:
        num_benefit += 1
    else:
        num_impair += 1

print("\nMODEL: o1-mini_max3x")
print("Dataset: ISOT")
print(f"Accuracy: {o_one_mini_max3x_num_correct/len(o_one_mini_max3x_data)}\n"
      f"Invalid Answers: {o_one_mini_max3x_num_invalid/len(o_one_mini_max3x_data)}")
print(f"Cleaned Accuracy: {o_one_mini_max3x_num_correct/(len(o_one_mini_max3x_data) - o_one_mini_max3x_num_invalid)}")
print(f"Accuracy (>341 tokens, benefit): {num_benefit_correct/num_benefit} ({num_benefit} instances)\n"
      f"Accuracy (<=341 tokens, impaired): {num_impair_correct/num_impair} ({num_impair} instances)")




##################################### dataset annotation with token num ################################################
# needs to be done once (adds the number of tokens per instance to the original dataset, necessary for o1-mini_max3x)
'''
# add column for number of tokens in instance
df['prompt_tokens'] = np.nan

# more complicated approach as jsonl file is not necessary in the correct order
for res in results:
    task_id = res['custom_id']
    idx = int(task_id.split('-')[-1])
    label = int(task_id.split('-')[-2])
    prompt_tokens = res['response']['body']['usage']['prompt_tokens']

    matches = (df['id'] == idx) & (df['label'] == label)

    if matches.any():
        df.loc[matches, 'prompt_tokens'] = prompt_tokens
    else:
        print(f"No match found for id: {idx}, label: {label}")

# Check if any cells weren't filled
print(df['prompt_tokens'].isna().any())
df = df[['id', 'label', 'prompt_tokens', 'title', 'text']]
df.to_csv('../LLM_data/ISOT_llm_test.csv')
'''


########################################################################################################################
############################################# Analysis for LIAR2 dataset ###############################################
########################################################################################################################

######################################## 4o-mini RESULTS ###############################################################


liar2_four_o_mini_file_name = '../LLM_data/res_LIAR2_gpt-4o-mini.csv'
liar2_four_o_mini_data = pd.read_csv(liar2_four_o_mini_file_name)
liar2_four_o_mini_num_correct = 0
liar2_four_o_mini_num_invalid = 0
#liar2_four_o_mini_num_valid = 0
for idx, row in liar2_four_o_mini_data.iterrows():
    result = row['prediction']
    label = row['label']
    if result == 0 or result == 1:
        #liar2_four_o_mini_num_valid += 1
        if int(result) == label:
            liar2_four_o_mini_num_correct += 1
    else:
        liar2_four_o_mini_num_invalid += 1

print("\nMODEL: GPT-4o-mini")
print("Dataset: LIAR2")
print(f"Accuracy: {liar2_four_o_mini_num_correct/len(liar2_four_o_mini_data)}\n"
      f"Invalid Answers: {liar2_four_o_mini_num_invalid/len(liar2_four_o_mini_data)}")
print(f"Cleaned Accuracy: {liar2_four_o_mini_num_correct/(len(liar2_four_o_mini_data) - liar2_four_o_mini_num_invalid)}")

############################################# 4o RESULTS ###############################################################


liar2_four_o_file_name = '../LLM_data/res_LIAR2_gpt-4o.csv'
liar2_four_o_data = pd.read_csv(liar2_four_o_file_name)
liar2_four_o_num_correct = 0
liar2_four_o_num_invalid = 0
#liar2_four_o_num_valid = 0
for idx, row in liar2_four_o_data.iterrows():
    result = row['prediction']
    label = row['label']
    if result == 0 or result == 1:
        #liar2_four_o_num_valid += 1
        if int(result) == label:
            liar2_four_o_num_correct += 1
    else:
        liar2_four_o_num_invalid += 1

print("\nMODEL: GPT-4o")
print("Dataset: LIAR2")
print(f"Accuracy: {liar2_four_o_num_correct/len(liar2_four_o_data)}\n"
      f"Invalid Answers: {liar2_four_o_num_invalid/len(liar2_four_o_data)}")
print(f"Cleaned Accuracy: {liar2_four_o_num_correct/(len(liar2_four_o_data) - liar2_four_o_num_invalid)}")


############################################# 4o RESULTS ###############################################################


liar2_o_one_file_name = '../LLM_data/res_LIAR2_o1-mini.csv'
liar2_o_one_data = pd.read_csv(liar2_o_one_file_name)
liar2_o_one_num_correct = 0
liar2_o_one_num_invalid = 0
#liar2_o_one_num_valid = 0
for idx, row in liar2_o_one_data.iterrows():
    result = row['prediction']
    label = row['label']
    if result == '0' or result == '1':
        #liar2_four_o_num_valid += 1
        if int(result) == label:
            liar2_o_one_num_correct += 1
    else:
        liar2_o_one_num_invalid += 1

print("\nMODEL: o1-mini")
print("Dataset: LIAR2")
print(f"Accuracy: {liar2_o_one_num_correct/len(liar2_o_one_data)}\n"
      f"Invalid Answers: {liar2_o_one_num_invalid/len(liar2_o_one_data)}")
print(f"Cleaned Accuracy: {liar2_o_one_num_correct/(len(liar2_o_one_data) - liar2_o_one_num_invalid)}")

nan_count = liar2_o_one_data['prediction'].isna().sum()
print(f"Number of empty prediction values: {nan_count}, "
      f"which makes {100*nan_count/liar2_o_one_num_invalid}% of all invalid responses")
