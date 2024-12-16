from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from AspireRAG import CareerRAG
import traceback 

app = Flask(__name__)

# 設定 CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://127.0.0.1:5500", "http://localhost:5500"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/career-advice', methods=['POST', 'OPTIONS'])
def get_career_advice():
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        return response
        
    try:
        if not request.is_json:
            return jsonify({'error': '請求格式必須是 JSON'}), 400
            
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': '請提供查詢內容'}), 400
            
        career_rag = CareerRAG()
        
        # 搜尋職缺
        job_type = career_rag.extract_job_type(query)
        if not job_type:
            job_type = '前端'  # 默認值
        
        jobs = career_rag.search_relevant_jobs(query)
        jobs_summary = career_rag.format_jobs_summary(jobs)
        
        skills = set()
        for job in jobs:
            if isinstance(job.get('擅長工具'), str):
                skills.update([s.strip() for s in job['擅長工具'].split(',')])
            if isinstance(job.get('工作技能'), str):
                skills.update([s.strip() for s in job['工作技能'].split(',')])
                
        # 傳入職位類型
        courses = career_rag.search_relevant_courses(list(skills), job_type)
        courses_summary = career_rag.format_courses_summary(courses)
        
        # 準備上下文
        context = f"""
        你現在是一位資深的職涯顧問，擅長分析產業趨勢和職涯規劃。
        你非常喜歡使用年輕人的語調去構築你的語句，善加利用表情符號語言文字，來體現你的親和力。
        你也擅長使用古典的名言佳句來增強自己的說服力道。
        分析職涯規劃之餘，你也會提供一些實用的人生歷練。
        請根據以下資訊，為對{query}感興趣的學生提供專業建議。

        請注意：
        1. 不要使用任何 Markdown 語法（如 ** 或 * 等符號）
        2. 不要使用任何特殊格式標記
        3. 永遠都使用繁體中文
        4. 可以適當使用表情符號語言文字，來體現你的親和力
        5. 不要使用*
        6. 不要使用**

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

        參考資料：
        {jobs_summary}
        {courses_summary}
        """
        
        # 生成建議
        response = career_rag.client.generate(model='llama3.2', prompt=context)
        advice = response['response']
        
        return jsonify({
            'status': 'success',
            'jobs': jobs_summary,
            'courses': courses_summary,
            'advice': advice
        })
        
    except Exception as e:
        error_traceback = traceback.format_exc()  # 獲取完整的錯誤堆疊
        print(f"錯誤詳情:\n{error_traceback}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)