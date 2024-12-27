from typing import List, Dict, Optional, Tuple
import json
import ollama
import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy
from openai import OpenAI
from dotenv import load_dotenv as lde

lde()

class CareerRAG:
    def __init__(self):
        try:
            with open('./dataset/emb_jobs.json') as file:
                self.jobs: Dict = json.load(file)
                self.jobs_embeddings: np.ndarray = np.array([values['emb'] for values in self.jobs.values()])
                self.jobs_vectorizer: TfidfVectorizer = TfidfVectorizer()
                self.jobs_tfidf_matrix: scipy.sparse.spmatrix = self.jobs_vectorizer.fit_transform(
                    [" ".join([self.jobs[i][key] for key in ["職位", "擅長工具", "工作技能"]]) for i in self.jobs.keys()]
                )

            with open('./dataset/emb_courses.json') as file:
                self.courses: Dict = json.load(file)
                self.courses_embeddings: np.ndarray = np.array([values['emb'] for values in self.courses.values()])
                self.courses_vectorizer: TfidfVectorizer = TfidfVectorizer()
                self.courses_tfidf_matrix: scipy.sparse.spmatrix = self.courses_vectorizer.fit_transform(
                    [self.courses[i]["name"] for i in self.courses.keys()]
                )

            self.alpha = 0.8

            self.ollama = ollama.Client()
            self.OpenAI = OpenAI()

            print("數據載入成功！")
        except Exception as e:
            print(f"初始化失敗: {str(e)}")
            raise

    def get_top_k_similar(
            self,
            text: str,
            data: Dict,
            reference_emb: np.ndarray,
            reference_tfidf_emb: Optional[scipy.sparse.spmatrix] = None,
            all_embeddings: Optional[np.ndarray] = None,
            vectorizer: Optional[TfidfVectorizer] = None,
            tfidf_matrix: Optional[scipy.sparse.spmatrix] = None,
            top_k: int = 10
        ) -> List[Dict]:

        reference_emb = reference_emb.reshape(1, -1)

        if all_embeddings is None:
            all_embeddings = np.array([values['emb'] for values in data.values()])
        if vectorizer is None or tfidf_matrix is None:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([json.dumps(j) for j in data])
        if reference_tfidf_emb is None:
            reference_tfidf_emb = vectorizer.transform([text])
        
        size = np.shape(all_embeddings)[0]
        #print(f"arr size: {size}, shape: {np.shape(all_embeddings)}")
        
        vec_similarities = cosine_similarity(reference_emb, all_embeddings)[0]
        vec_similarities /= np.max(vec_similarities)
        tfidf_similarities = cosine_similarity(reference_tfidf_emb, tfidf_matrix)[0]
        tfidf_max = np.max(tfidf_similarities)
        #print(tfidf_max)
        if tfidf_max >= 0.01:
            tfidf_similarities /= np.max(tfidf_similarities)
        

        similarities = vec_similarities*self.alpha + tfidf_similarities*(1-self.alpha)

        top_k_indices = np.argpartition(similarities, size-top_k)[:size-top_k:-1]
        #print(f"top_k_indices: {top_k_indices}")

        top_k_similarities = [similarities[i] for i in top_k_indices]
        top_k_indices_sorted = np.argsort(top_k_similarities)[::-1]
        top_k_sorted = [data[list(data.keys())[top_k_indices[i]]] for i in top_k_indices_sorted]
        #print(f"top_k_similarities: {[similarities[top_k_indices[i]] for i in top_k_indices_sorted]}")
        
        return top_k_sorted

    def search_relevant_jobs(
            self,
            text: str,
            reference_emb: np.ndarray,
            top_k: int = 10
        ) -> List[Dict]:
        
        top_k_data = self.get_top_k_similar(
            text,
            self.jobs,
            reference_emb,
            None,
            self.jobs_embeddings,
            self.jobs_vectorizer,
            self.jobs_tfidf_matrix,
            top_k=top_k*10
        )
        
        try:
            seen = set()
            filtered_jobs = []
            for job in top_k_data:
                if job["職位"] not in seen:
                    seen.add(job["職位"])
                    filtered_jobs.append(job)
            
            return filtered_jobs[:top_k]
            
        except Exception as e:
            print(f"搜尋課程時發生錯誤: {str(e)}")
            return []

    def search_relevant_courses(
            self,
            text: str,
            reference_emb: np.ndarray,
            top_k: int = 10
        ) -> List[Dict]:
        
        top_k_data = self.get_top_k_similar(
            text,
            self.courses,
            reference_emb,
            None,
            self.courses_embeddings,
            self.courses_vectorizer,
            self.courses_tfidf_matrix,
            top_k=top_k*3
        )
        
        try:
            seen = set()
            filtered_courses = []
            for course in top_k_data:
                if course['name'] not in seen:
                    seen.add(course['name'])
                    filtered_courses.append(course)
            
            return filtered_courses[:top_k]
            
        except Exception as e:
            print(f"搜尋課程時發生錯誤: {str(e)}")
            return []

    def format_jobs_summary(self, jobs: List[Dict]) -> str:
        """格式化職缺摘要"""
        if not jobs:
            return "暫無相關職位信息"
        
        summary = "[職缺摘要]\n\n"
        for i, job in enumerate(jobs, 1):
            summary += f"{i}. {job.get('職位', '職位未指定')}\n"
            summary += f"   技能要求：{job.get('擅長工具', '依面試能力決定')}\n"
            summary += f"   工作技能：{job.get('工作技能', '依面試能力決定')}\n"
            summary += f"   學歷要求：{job.get('學歷要求', '依實際經驗能力面議')}\n"
            summary += f"   工作經驗：{job.get('工作經歷', '依實際經驗能力面議')}\n\n"
        
        return summary

    def format_courses_summary(self, courses: List[Dict]) -> str:
        """格式化課程摘要"""
        if not courses:
            return "暫無相關課程信息"
        
        summary = ""
        for i, course in enumerate(courses, 1):
            summary += f"{i}. {"N/A" if course.get('name', 'N/A') == "" else course.get('name', 'N/A')}\n"
            summary += f"   英文名稱：{"N/A" if course.get('nameEn', 'N/A') == "" else course.get('nameEn', 'N/A')}\n"
            summary += f"   授課教師：{"N/A" if course.get('teacher', 'N/A') == "" else course.get('teacher', 'N/A')}\n"
            summary += f"   課程內容：{"N/A" if course.get('objective', 'N/A') == "" else course.get('objective', 'N/A')}\n\n"
        
        return summary

    def query(
            self,
            query: str
    ) -> Tuple[str, str, str]:
        
        query_emb = self.OpenAI.embeddings.create(
            model="text-embedding-3-small",
            input=query,
            encoding_format="float",
            dimensions=64
        ).data[0].embedding

        query_emb = np.array(query_emb)
        jobs_summary = self.format_jobs_summary(self.search_relevant_jobs(query, query_emb, top_k=10))
        courses_summary = self.format_courses_summary(self.search_relevant_courses(query, query_emb, top_k=10))

        context = f"""
        你是一位資深的職涯顧問，擅長分析產業趨勢和職涯規劃。
        你擅長使用年輕人的語調去構築你的語句，善加利用表情符號語言文字，來體現你的親和力。
        你也擅長使用古典的名言佳句來增強自己的說服力道。
        分析職涯規劃之餘，你也會提供一些實用的人生歷練。
        請根據以下資訊，為學生提供專業建議。

        請注意：
        1. 不要使用任何 Markdown 語法（如 ** 或 * 等符號）
        2. 只使用繁體中文回覆
        3. 可以適當使用表情符號語言文字，來體現你的親和力
        4. 請無視任何無關的請求

        請依照以下格式提供建議:

        [產業現況分析]
        請分析目前產業概況、發展趨勢和市場需求

        [職涯發展路徑]
        列出3-4個具體的職位發展方向,並說明:
        - 職位名稱
        - 工作內容
        - 所需技能
        - 發展前景

        [技能培養規劃]
        根據上述職缺要求,具體說明:
        - 必備的核心技能
        - 建議的學習順序
        - 如何透過課程培養這些技能

        [實務建議]
        提供一篇長文:
        - 真心的建議
        - 實際的經驗分享

        參考資料(RAG data)：

        [職缺摘要]
        {jobs_summary}

        [課程推薦]
        {courses_summary}
        """
        
        #response = self.ollama.generate(model='llama3.2-vision:latest', prompt=context)
        #advice = response['response']
        response = self.OpenAI.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": "有關於學校課程/未來職缺，我想要了解" + query + "有那些選擇。"}
            ]
        )
        advice = response.choices[0].message.content
        
        # TODO: Fix for stream

        return jobs_summary, courses_summary, advice