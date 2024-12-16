import pandas as pd
from openai import OpenAI
import json

import datasets

########################################################################################################################
# This file contains the calls to the OpenAI API
# It runs only when the OpenAI API Key is set as environmental variable on your local machine
# All calls to the API are currently as comments, to use them
# - Batch Processing: Create Batch file via section BATCH CREATION & UPLOAD then analyse result (after processing) via
#   section RETRIEVING RESULTS & PROCESSING
# - Manual Processing (required for some models): use code from section MANUAL PROCESSING FOR MODELS WITHOUT
#   BATCH PROCESSING to perform calls with the ISOT dataset and use code from section API Calls FOR LIAR2 DATASET to
#   perform calls with the LIAR2 dataset (change desired model, and correct temperature/max_completion_tokens parameters
#   as necessary)
# Be careful, as the use of the API isn't free!
########################################################################################################################



########################################################################################################################
############################################# API Calls for ISOT dataset ###############################################
########################################################################################################################

def prepare_data():
    # Reconstruct unprocessed versions of articles from testset
    test_data = pd.read_csv("../dataset/test.csv")
    true_data = pd.read_csv("../dataset/True.csv")
    fake_data = pd.read_csv("../dataset/Fake.csv")

    # Split test data to in real and fake
    test_real = test_data[test_data['label'] == 1]
    test_fake = test_data[test_data['label'] == 0]

    # Add column id to original ISOT dataset
    true_data['id'] = range(0, len(true_data))
    fake_data['id'] = range(0, len(fake_data))

    # Merge with original datasets to obtain unprocessed title and body
    merged_real = pd.merge(test_real[['id', 'label']], true_data[['id', 'title', 'text']], on='id', how='left')
    merged_fake = pd.merge(test_fake[['id', 'label']], fake_data[['id', 'title', 'text']], on='id', how='left')

    # Combine to complete dataset
    unprocessed_data = pd.concat([merged_real, merged_fake], ignore_index=True)
    unprocessed_data = unprocessed_data[['id', 'label', 'title', 'text']]
    unprocessed_data.to_csv('../LLM_data/ISOT_llm_test.csv', index=False)


######################################## BATCH CREATION & UPLOAD ######################################################


# get data
#prepare_data() # Needs to be done 1 time
data = pd.read_csv("../LLM_data/ISOT_llm_test.csv")

# Initializing OpenAI client
client = OpenAI()

'''
# creating batch file for batch processing of prompts
tasks = []

for index, row in data.iterrows():
    text = row['text']
    title = row['title']
    num = row['id']
    label = row['label']
    prompt = ("I want you to rate the truthfulness of the following news article. "
              "Do not provide any explanations or additional information, just answer with 0 or 1, "
              "where 0 represents false and 1 represents true. \n \n"
              "Title: " + title + "\nText: " + text)

    task = {
        "custom_id": f"task-{label}-{num}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            #"max_completion_tokens": 1024, # for o1-mini model 
            # response_format={
            #    "type": "json_object"
            # },
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        }
    }

    tasks.append(task)

# creating the file
file_name = "../LLM_data/batch_tasks_ISOT_gpt-4o-mini.jsonl"
with open(file_name, "w") as file:
    for obj in tasks:
        file.write(json.dumps(obj) + "\n")

# upload file
batch_file = client.files.create(
    file=open(file_name, "rb"),
    purpose="batch"
)
print(batch_file)


# creating batch job
batch_job = client.batches.create(
  input_file_id=batch_file.id,
  endpoint="/v1/chat/completions",
  completion_window="24h"
)

# checking batch status
batch_job = client.batches.retrieve(batch_job.id)
#batch_job = client.batches.retrieve('batch_675f17692a8081908ff134e7ed370be9')
print(batch_job)
'''

#################################### RETRIEVING RESULTS & PROCESSING ##################################################


'''
# retrieving results and save in file
result_file_id = batch_job.output_file_id
result = client.files.content(result_file_id).content

result_file_name = '../LLM_data/res_ISOT_gpt-4o-mini.jsonl'

with open(result_file_name, 'wb') as file:
    file.write(result)

# Loading data from saved file
results = []
with open(result_file_name, 'r') as file:
    for line in file:
        # Parsing the JSON string into a dict and appending to the list of results
        json_object = json.loads(line.strip())
        results.append(json_object)


# Reading the results
num_correct = 0 # Instances answered correctly
num_invalid = 0 # Instances with answers other than 0 or 1
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
            num_correct += 1
    else:
        num_invalid += 1


print(f"Accuracy: {num_correct/len(results)}\nInvalid Answers: {num_invalid}")
print(f"Total Input Tokens: {total_prompt_tokens}")
'''


############################# MANUAL PROCESSING FOR MODELS WITHOUT BATCH PROCESSING ####################################

# Getting LLM answer for single examples
def get_prediction(title, text, dataset, prompt_tokens=1):
    if dataset == 'ISOT':
        prompt = ("I want you to rate the truthfulness of the following news article. "
                  "Do not provide any explanations, just answer with 0 or 1, "
                  "where 0 represents false and 1 represents true. \n \n"
                  "Title: " + title + "\nText: " + text)
    elif dataset == 'LIAR2':
        prompt = ("I want you to rate the truthfulness of the following statement. "
                  "Do not provide any explanations, just answer with 0 or 1, "
                  "where 0 represents false and 1 represents true. \n \n"
                  "Statement: " + text)
    else:
        raise ValueError("Invalid Dataset")

    response = client.chat.completions.create(
        model="o1-mini",                    # models to use here: gpt-4o, gpt-4o-mini, o1-mini
        #temperature=0.0,                    # temperature settings unavailable for o1-mini (default value: 1)
        #max_completion_tokens=1024,                    # use with ISOT Dataset
        #max_completion_tokens=int(3*prompt_tokens),    # use for max3x version
        max_completion_tokens=512,                      # use with LIAR2 Dataset
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

    )

    return response.choices[0].message.content

'''
results = pd.DataFrame(columns=['id', 'label', 'prediction', 'title', 'text'])
for idx, row in data.iterrows():
    text = row['text']
    title = row['title']
    prompt_tokens = row['prompt_tokens']
    pred = get_prediction(title, text, 'ISOT', prompt_tokens)
    label =row['label']
    id = row['id']
    print(f"TITLE: {title}\nTEXT: {text}\n\nRESULT: {pred}\nTRUE LABEL:{label}")
    print("\n\n----------------------------\n\n")
    out = {'id': id, 'label': label, 'prediction': pred, 'title': title, 'text': text}
    results.loc[idx] = out

output_filename = '../LLM_data/res_ISOT_o1-mini_max3x.csv' # change model name as necessary
results.to_csv(output_filename, index=False)

'''




########################################################################################################################
############################################ API CALLS FOR LIAR2 DATASET ###############################################
########################################################################################################################
'''
dataset_name = "chengxuphd/liar2"
dataset = datasets.load_dataset(dataset_name)
liar2 = pd.DataFrame(columns=['id', 'label', 'prediction', 'statement'])
liar2['id'] = dataset["test"]["id"]
liar2['label'] = dataset["test"]["label"]
liar2['statement'] = dataset["test"]["statement"]
# Collapse categories to true (1) and false (0)
liar2['label'] = liar2['label'].apply(lambda x: 0 if x <= 2 else 1)


for idx, row in liar2.iterrows():
    statement = row['statement']
    label = row['label']
    try:
        pred = get_prediction("", statement, 'LIAR2')   #prompt_tokens not needed bc statements are shorter
    except Exception:
        pred = "FAILED"

    print(f"STATEMENT: {statement}\n\nRESULT: {pred}\nTRUE LABEL:{label}")
    print("\n\n----------------------------\n\n")
    liar2.loc[idx, 'prediction'] = pred

output_filename = '../LLM_data/res_LIAR2_o1-mini.csv' # change model name as necessary
liar2.to_csv(output_filename, index=False)

'''