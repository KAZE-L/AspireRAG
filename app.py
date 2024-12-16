from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from AspireRAG import CareerRAG

app = Flask(__name__)

# 設定 CORS，允許特定來源
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
    # 處理 OPTIONS 請求
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
        
        # 獲取職缺和課程資訊
        jobs = career_rag.search_relevant_jobs(query)
        jobs_summary = career_rag._format_jobs_summary(jobs)
        
        # 獲取相關技能
        skills = set()
        for job in jobs:
            if isinstance(job.get('擅長工具'), str):
                skills.update([s.strip() for s in job['擅長工具'].split(',')])
            if isinstance(job.get('工作技能'), str):
                skills.update([s.strip() for s in job['工作技能'].split(',')])
                
        courses = career_rag.search_relevant_courses(list(skills))
        courses_summary = career_rag._format_courses_summary(courses, list(skills))
        
        # 準備上下文資訊給 LLM
        context = f"""
        你現在是一位充滿智慧的大學教授，正在為學生提供職涯諮詢。
        這位學生對{query}感興趣。
        請以關心且睿智的語氣，運用文言文風格，為這位學生提供建議。
        不需要重複列出職缺和課程資訊，只需提供個人化的建議即可。
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
        print(f"Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)