from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from AspireRAG import CareerRAG
import traceback
import os

template_dir = os.path.abspath('./template')
static_dir = os.path.abspath('./static')
app = Flask(
        __name__,
        template_folder=template_dir,
        static_folder=static_dir
)

print("啟動中...")
__career_rag = CareerRAG()

# 設定 CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://aspire.ebg.tw/"],
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
        
        jobs_summary, courses_summary, advice = __career_rag.query(query)
        
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