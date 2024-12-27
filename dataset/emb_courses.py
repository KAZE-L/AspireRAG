import sqlite3
import json
from pprint import pprint
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv as lde

batch_size = 256
lde()

client = OpenAI()

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def build(dataFile="NCCUcourse.db"):
    con = sqlite3.connect(dataFile)
    con.row_factory = dict_factory
    cursor = con.cursor()

    req = cursor.execute("SELECT *, max(`id`) FROM COURSE GROUP BY subNum")
    res = req.fetchall()

    lst = [
        {
            'unit': d['unit'],
            'unitEn': d['unitEn'],
            'name': d['name'],
            'nameEn': d['nameEn'],
            'note': d['note'],
            'noteEn': d['noteEn'],
            'objective': d['objective'],
            'syllabus': d['syllabus'],
            'unit': d['unit'],
            'unitEn': d['unitEn'],
            'teacher': d['teacher']
        }
        for d in res
    ]

    emb_dict = {}
    batches = [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]
    for i in tqdm(range(len(batches))):
        batch = batches[i]
        lines = []
        skipped = 0
        for j in range(len(batch)):
            data = batch[j]
            line = json.dumps(data, ensure_ascii=False)
            if(len(line) <= 6000):
                lines.append(line)
                #print(len(line), end=' ')
                index = i*batch_size + j - skipped
                emb_dict[f"request-{index:05d}"] = data
            else:
                skipped += 1
        embs = client.embeddings.create(
            model="text-embedding-3-small",
            input=lines,
            encoding_format="float",
            dimensions=64,
        )

        for emb_item in embs.data:
            index = i*batch_size + emb_item.index
            emb_dict[f"request-{index:05d}"]["emb"] = emb_item.embedding

    embfile = open("emb_courses.json", "w")
    print(json.dumps(emb_dict, ensure_ascii=False), file=embfile)

if __name__ == "__main__":
    build()