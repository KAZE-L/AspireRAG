# AspireRAG - 職涯諮詢助手

一個基於 RAG 技術的職涯諮詢系統，幫助使用者探索職業發展方向。

## 功能特點

- 職位分析：提供詳細的職位資訊和市場需求
- 課程建議：推薦相關的學習課程
- 職涯建議：提供專業的職涯規劃建議

## Build and run

### Step 1. Install requirements

```python
python3 -m pip install -r requirements.txt
```

### Step 2. Modify .env

```env
OPENAI_API_KEY="sk-proj-YoUr_OpEnAi_ApIkEy"
```

### Step 3. Calculate embeddings (via OpenAI API)

```bash
cd dataset
python3 ./emb_jobs.py
python3 ./emb_courses.py
```

### Step 4. Run development

```bash
./run_dev.sh
```

And you're good to go.