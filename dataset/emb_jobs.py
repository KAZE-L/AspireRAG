import csv
import json
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv as lde

batch_size = 256
lde()

client = OpenAI()

with open('rawdata_jobs.csv', 'r', newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    headers = next(reader, None)
    outfile = open("data.jsonl", "w")
    emb_dict = {}
    lst = list(reader)
    batches = [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]
    for i in tqdm(range(len(batches))):
        batch = batches[i]
        lines = []
        for j in range(len(batch)): 
            row = [l.replace("\n", ' ') for l in batch[j]]
            line = ", ".join([f"{key}: {val}" for key, val in zip(headers, row)])
            lines.append(line)

            data = {}
            for key, val in zip(headers, row):
                data[key] = val
            
            index = i*batch_size + j
            emb_dict[f"request-{index:05d}"] = data

        embs = client.embeddings.create(
            model="text-embedding-3-small",
            input=lines,
            encoding_format="float",
            dimensions=64,
        )

        for emb_item in embs.data:
            
            index = i*batch_size + emb_item.index
            emb_dict[f"request-{index:05d}"]["emb"] = emb_item.embedding   

    embfile = open("emb_work.json", "w")
    print(json.dumps(emb_dict, ensure_ascii=False), file=embfile)