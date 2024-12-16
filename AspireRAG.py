import pandas as pd
from typing import List, Dict
import ollama

class CareerRAG:
    def __init__(self):
        """初始化職涯諮詢系統"""
        try:
            self.jobs_df = pd.read_csv('職稱列表.csv')
            self.courses_df = pd.read_csv('課程列表.csv')
            self.client = ollama.Client()
            print("數據載入成功！")
        except Exception as e:
            print(f"初始化失敗: {str(e)}")
            raise

    def extract_job_type(self, text: str) -> str:
        """從自然語言中提取職位類型"""
        keywords = {
            '前端': [
                '前端', '網頁', '前台', '網站'
                '使用者介面', '網站設計', '網頁設計',
                '使用者體驗', '網站開發', '網頁開發',
                '介面設計', '視覺設計', '互動設計',
                '網頁美工', '網頁製作', '網站製作',
                '前台開發', '瀏覽器開發', '網頁程式',
                '使用者互動', '響應式設計', '網頁架構',
                '前端互動', '網頁效能', '網站效能',
                '前端優化', '網頁優化', '網站優化',
                '前端維護', '網頁維護', '網站維護'
            ],
            '後端': [
                '後端', '後台', '伺服器', 
                '資料庫', '系統開發', '系統架構',
                '資料處理', '程式邏輯', '商業邏輯',
                '後台管理', '系統管理', '資料管理',
                '伺服器維護', '系統維護', '資料維護',
                '系統整合', '資料整合', '系統分析',
                '效能調校', '系統調校', '資料調校',
                '資安防護', '系統安全', '資料安全',
                '系統監控', '效能監控', '資源監控',
                '系統擴充', '效能優化', '架構優化'
            ],
            '全端': [
                '全端', '全棧', '全方位',
                '全站開發', '全方位開發', '全端開發',
                '系統全端', '網站全端', '應用全端',
                '全方位系統', '全端架構', '全棧架構',
                '全端維護', '全站維護', '全方位維護',
                '全端整合', '全站整合', '全方位整合',
                '全端規劃', '全站規劃', '全方位規劃',
                '全端設計', '全站設計', '全方位設計',
                '全端優化', '全站優化', '全方位優化',
                '全端管理', '全站管理', '全方位管理'
            ]
        }
        
        # 轉換為小寫以進行比對
        text = text.lower()
        
        # 檢查每個職位類型的關鍵字
        for job_type, type_keywords in keywords.items():
            if any(keyword in text for keyword in type_keywords):
                print(f"匹配到職位類型: {job_type}")  # 調試信息
                return job_type
        
        print("未匹配到職位類型，使用默認值: 前端")  # 調試信息
        return None

    def search_relevant_jobs(self, text: str) -> List[Dict]:
        """搜尋相關職位"""
        try:
            # 從輸入文字中提取職位類型
            job_type = self.extract_job_type(text)
            if not job_type:
                print("無法識別職位類型")
                return []
                
            print(f"識別到的職位類型：{job_type}")
            
            # 定義職位關鍵字映射
            job_keywords = {
            '前端': {
                '職稱關鍵字': {
                    '前端工程師': 20,
                    '網頁工程師': 15,
                    '全端工程師': 10,
                    'Frontend Engineer': 20,
                    'Web Developer': 15,
                    'UI工程師': 15,
                    '前端開發工程師': 20,
                    '網頁設計師': 12,
                    '前端': 20,
                    '網頁': 15,
                    '全端': 10,
                    'frontend': 20,
                    'web': 15
                },
                '技能關鍵字': {
                    'html': 5,
                    'css': 5,
                    'javascript': 5,
                    'react': 4,
                    'vue': 4,
                    'angular': 4,
                    'jquery': 3,
                    'typescript': 3,
                    'nodejs': 2,
                    'webpack': 3,
                    'sass': 3,
                    'less': 3,
                    'bootstrap': 3,
                    'tailwind': 3,
                    'redux': 3,
                    'next.js': 4,
                    'nuxt.js': 4,
                    'responsive': 3,
                    'web design': 3,
                    'ui/ux': 3,
                    'git': 2,
                    'npm': 2,
                    'yarn': 2,
                    'rest api': 2,
                    'graphql': 2
                }
            },
            '後端': {
                '職稱關鍵字': {
                    '後端工程師': 20,
                    '伺服器工程師': 15,
                    '全端工程師': 10,
                    'Backend Engineer': 20,
                    'Server Developer': 15,
                    '系統工程師': 12,
                    '後端開發工程師': 20,
                    '資料庫工程師': 12,
                    '後端': 20,
                    '伺服器': 15,
                    '全端': 10,
                    'backend': 20,
                    'server': 15
                },
                '技能關鍵字': {
                    'java': 5,
                    'python': 5,
                    'nodejs': 5,
                    'php': 4,
                    'c#': 4,
                    '.net': 4,
                    'sql': 4,
                    'mysql': 4,
                    'postgresql': 4,
                    'mongodb': 4,
                    'redis': 3,
                    'spring': 4,
                    'django': 4,
                    'laravel': 4,
                    'express': 4,
                    'restful': 3,
                    'api': 3,
                    'docker': 3,
                    'kubernetes': 3,
                    'aws': 3,
                    'azure': 3,
                    'linux': 3,
                    'git': 2,
                    'ci/cd': 2,
                    'microservices': 2
                }
            },
            '全端': {
                '職稱關鍵字': {
                    '全端工程師': 20,
                    '全棧工程師': 20,
                    'Fullstack Engineer': 20,
                    '全端開發工程師': 20,
                    '全端': 20,
                    '全棧': 20,
                    'fullstack': 20
                },
                '技能關鍵字': {
                    'html': 4,
                    'css': 4,
                    'javascript': 4,
                    'react': 4,
                    'vue': 4,
                    'angular': 4,
                    'nodejs': 4,
                    'java': 4,
                    'python': 4,
                    'php': 4,
                    'sql': 4,
                    'mongodb': 4,
                    'redis': 3,
                    'docker': 3,
                    'kubernetes': 3,
                    'aws': 3,
                    'git': 3,
                    'ci/cd': 3,
                    'restful': 3,
                    'api': 3,
                    'typescript': 3,
                    'webpack': 3,
                    'spring': 3,
                    'django': 3,
                    'laravel': 3
                }
            }
        }
            
            keywords = job_keywords.get(job_type, {
                '職稱關鍵字': {job_type: 20},
                '技能關鍵字': {}
            })
            
            relevant_jobs = []
            
            for _, job in self.jobs_df.iterrows():
                score = 0
                
                # 檢查職稱相關性
                job_title = str(job['職稱']).lower()
                # 先檢查完全匹配
                exact_match = False
                for title, weight in keywords['職稱關鍵字'].items():
                    if title.lower() == job_title:
                        score += weight
                        exact_match = True
                        break
                
                # 如果沒有完全匹配，再檢查部分匹配
                if not exact_match:
                    for title, weight in keywords['職稱關鍵字'].items():
                        if title.lower() in job_title:
                            score += weight
                            break
                
                # 檢查技能相關性
                if not pd.isna(job['擅長工具']):
                    tools = str(job['擅長工具']).lower()
                    for skill, weight in keywords['技能關鍵字'].items():
                        if skill.lower() in tools:
                            score += weight
                
                if not pd.isna(job['工作技能']):
                    skills = str(job['工作技能']).lower()
                    for skill, weight in keywords['技能關鍵字'].items():
                        if skill.lower() in skills:
                            score += weight // 2
                
                # 如果具有前端核心技能，確保至少有基礎分數
                core_skills = ['html', 'css', 'javascript']
                if any(skill in str(job['擅長工具']).lower() for skill in core_skills):
                    score = max(score, 5)
                
                # 只要有分數就加入結果
                if score > 0:
                    job_dict = job.to_dict()
                    job_dict['相關度'] = score
                    relevant_jobs.append(job_dict)
            
            # 按相關度排序
            sorted_jobs = sorted(relevant_jobs, key=lambda x: x['相關度'], reverse=True)
            
            # 印出搜尋結果（用於調試）
            print("\n找到的職位：")
            for job in sorted_jobs[:10]:
                print(f"職稱: {job['職稱']}")
                print(f"技能: {job['擅長工具']}")
                print(f"相關度: {job['相關度']}\n")
            
            return sorted_jobs[:10]
            
        except Exception as e:
            print(f"搜尋職位時發生錯誤: {str(e)}")
            return []

    def search_relevant_courses(self, skills: List[str], job_type: str = '前端') -> List[Dict]:
        """搜尋相關課程"""
        try:
            print(f"當前搜尋的職位類型: {job_type}")  # 調試信息
            
            # 定義不同職位類型的關鍵字和權重
            course_keywords = {
                '前端': {
                    'javascript': 20,
                    'html': 10,
                    'css': 10,
                    'react': 8,
                    'vue': 8,
                    'angular': 8,
                    '前端': 8,
                    'frontend': 8,
                    'web': 20,
                    '程式': 8,
                    '使用者介面': 7,
                    '網頁設計': 7,
                    'typescript': 6,
                    'jquery': 5,
                    'bootstrap': 5,
                    'sass': 5,
                    'webpack': 5
                },
                '後端': {
                    'java': 10,
                    'python': 10,
                    'nodejs': 10,
                    'php': 8,
                    'sql': 8,
                    'database': 8,
                    '資料庫': 8,
                    '後端': 8,
                    'backend': 8,
                    'spring': 7,
                    'django': 7,
                    'mongodb': 7,
                    'redis': 6,
                    'api': 6,
                    '伺服器': 6,
                    'server': 6,
                    'linux': 5,
                    'docker': 5
                },
                '全端': {
                    '程式': 20,
                    '全端': 10,
                    '全棧': 10,
                    'javascript': 8,
                    'python': 8,
                    'java': 8,
                    'html': 8,
                    'css': 8,
                    'sql': 8,
                    'nodejs': 7,
                    'react': 7,
                    'vue': 7,
                    'spring': 7,
                    'django': 7,
                    'database': 7,
                    '資料庫': 7,
                    'api': 6,
                    'docker': 5
                }
            }

            # 根據職位類型選擇對應的關鍵字權重
            keywords = course_keywords.get(job_type, course_keywords['前端'])
            print(f"使用的關鍵字權重: {keywords}")  # 調試信息
            
            relevant_courses = []
            
            # 遍歷所有課程
            for _, course in self.courses_df.iterrows():
                score = 0
                course_text = f"{str(course['course_name_zh'])} {str(course['course_name_en'])} {str(course['notes'])}".lower()
                
                # 計算關鍵字分數
                for keyword, weight in keywords.items():
                    if keyword.lower() in course_text:
                        score += weight
                        print(f"課程 '{course['course_name_zh']}' 匹配到關鍵字 '{keyword}', 加分: {weight}")  # 調試信息
                
                # 檢查技能關鍵字
                for skill in skills:
                    if skill.lower() in course_text:
                        score += 2
                        print(f"課程 '{course['course_name_zh']}' 匹配到技能 '{skill}', 加分: 2")  # 調試信息
                
                # 只添加有分數的課程
                if score > 0:
                    relevant_courses.append({
                        '課程名稱': course['course_name_zh'],
                        '英文名稱': course['course_name_en'],
                        '學分數': course['credits'],
                        '授課教師': course['instructor_zh'],
                        '上課時間': course['time'],
                        '上課地點': course['location'],
                        '課程內容': course['notes'],
                        '相關度': score
                    })
            
            # 根據相關度排序
            sorted_courses = sorted(relevant_courses, key=lambda x: x['相關度'], reverse=True)
            
            # 去重
            seen = set()
            filtered_courses = []
            for course in sorted_courses:
                if course['課程名稱'] not in seen:
                    seen.add(course['課程名稱'])
                    filtered_courses.append(course)
            
            print(f"\n找到 {len(filtered_courses)} 門相關課程")  # 調試信息
            for course in filtered_courses[:3]:
                print(f"課程: {course['課程名稱']}, 相關度: {course['相關度']}")  # 調試信息
            
            return filtered_courses[:10]
            
        except Exception as e:
            print(f"搜尋課程時發生錯誤: {str(e)}")
            return []

    def format_jobs_summary(self, jobs: List[Dict]) -> str:
        """格式化職缺摘要"""
        if not jobs:
            return "暫無相關職位信息"
        
        summary = "[職缺摘要]\n\n"
        for i, job in enumerate(jobs[:30], 1):
            summary += f"{i}. {job.get('職稱', '職稱未指定')}\n"
            summary += f"   技能要求：{job.get('擅長工具', '依面試能力決定')}\n"
            summary += f"   工作技能：{job.get('工作技能', '依面試能力決定')}\n"
            summary += f"   學歷要求：{job.get('學歷要求', '依實際經驗能力面議')}\n"
            summary += f"   工作經驗：{job.get('工作經歷', '依實際經驗能力面議')}\n\n"
        
        return summary

    def format_courses_summary(self, courses: List[Dict]) -> str:
        """格式化課程摘要"""
        if not courses:
            return "暫無相關課程信息"
        
        summary = "[課程推薦]\n\n"
        for i, course in enumerate(courses[:20], 1):
            summary += f"{i}. {course.get('課程名稱', '未指定')}\n"
            summary += f"   英文名稱：{course.get('英文名稱', '未指定')}\n"
            summary += f"   授課教師：{course.get('授課教師', '未指定')}\n"
            summary += f"   課程內容：{course.get('課程內容', '未指定')}\n\n"
        
        return summary

    def generate_career_advice(self, query: str) -> str:
        """生成職涯建議"""
        try:
            # 先提取職位類型
            job_type = self.extract_job_type(query)
            if not job_type:
                job_type = '前端'  # 默認值
            
            jobs = self.search_relevant_jobs(query)
            jobs_summary = self.format_jobs_summary(jobs)
            
            skills = set()
            for job in jobs:
                if isinstance(job.get('擅長工具'), str):
                    skills.update([s.strip() for s in job['擅長工具'].split(',')])
                if isinstance(job.get('工作技能'), str):
                    skills.update([s.strip() for s in job['工作技能'].split(',')])
                    
            # 傳入職位類型
            courses = self.search_relevant_courses(list(skills), job_type)
            courses_summary = self.format_courses_summary(courses)
            
            context = f"""
            你現在是一位資深的職涯顧問教授，擅長分析產業趨勢和職涯規劃。
            你非常喜歡使用年輕人的語調去構築你的語句。
            分析職涯規劃之餘，你也會提供一些實用的人生歷練。 
            你會使用表情符號語言文字，來體現你的親和力。
            你也擅長使用古典的名言佳句來增強自己的說服力道。

            請注意：
            1. 不要使用任何 Markdown 語法（如 ** 或 * 等符號）
            2. 不要使用任何特殊格式標記

            請根據以下資訊，為對{query}感興趣的學生提供專業建議。

            [整體產業趨勢與建議]
            請用300-500字完整分析：
            1. 產業現況與未來發展
            2. 市場需求與薪資展望
            3. 技術發展趨勢
            4. 職涯規劃建議
            5. 學習路徑推薦

            [產業現況分析]
            請條列式說明：
            1. 目前產業概況
            2. 發展趨勢
            3. 市場需求

            [職涯發展路徑]
            請列出3-4個具體的職位發展方向：
            1. 職位一
            - 職位名稱：
            - 工作內容：
            - 所需技能：
            - 發展前景：

            2. 職位二
            - 職位名稱：
            - 工作內容：
            - 所需技能：
            - 發展前景：

            3. 職位三
            - 職位名稱：
            - 工作內容：
            - 所需技能：
            - 發展前景：

            [技能培養規劃]
            請條列式說明：
            1. 必備的核心技能：
            2. 建議的學習順序：
            3. 如何透過課程培養這些技能：

            [實務建議與心得分享]
            請用300-500字分享：
            1. 求職準備建議
            2. 面試技巧重點
            3. 職場生存之道
            4. 自我提升方向
            5. 人生經驗分享

            參考資料：
            {jobs_summary}
            {courses_summary}
            """
            
            response = self.client.chat(
                model='llama3.2',
                messages=[{
                    'role': 'user', 
                    'content': context
                }],
                options={
                    'temperature': 0.3,
                    'top_p': 0.2,
                    'frequency_penalty': 0.0,
                    'presence_penalty': 0.0,
                    'max_tokens': 2000  # 增加長度限制
                }
            )
            return f"{jobs_summary}\n{courses_summary}\n\n{response['response']}"
            
        except Exception as e:
            print(f"生成建議時發生錯誤: {str(e)}")
            return "無法生成完整建議"

def main():
    """主程序"""
    try:
        advisor = CareerRAG()
        print("初始化完成！歡迎使用職涯諮詢系統")
        
        while True:
            query = input("\n請描述你想了解的職位 (輸入 'quit' 結束): ").strip()
            
            if query.lower() == 'quit':
                print("\n感謝使用職涯諮詢系統！")
                break
                
            if not query:
                print("請輸入有效的查詢內容")
                continue
                
            response = advisor.generate_career_advice(query)
            print("\n" + response + "\n")
            
    except Exception as e:
        print(f"系統執行錯誤: {str(e)}")

if __name__ == "__main__":
    main()