import pandas as pd
from typing import List, Dict
import ollama
import time
import os

class CareerRAG:
    def __init__(self):
        """初始化職涯諮詢系統"""
        try:
            # 載入資料
            self.jobs_df = pd.read_csv('職稱列表.csv')
            self.courses_df = pd.read_csv('課程列表.csv')
            self.client = ollama.Client()
            print("數據載入成功！")
        except Exception as e:
            print(f"初始化失敗: {str(e)}")
            raise
        
    def search_relevant_jobs(self, job_type: str) -> List[Dict]:
        """搜尋相關職位"""
        try:
            # 定義工程師類型映射
            engineer_types = {
                # 網頁相關
                '前端': ['前端工程師', '網頁工程師', '.NET工程師', 'Node.js工程師'],
                '後端': ['後端工程師', 'Java工程師', 'Python工程師', '.NET工程師'],
                '全端': ['全端工程師', '資深全端工程師'],
                
                # 資料相關
                '資料': ['資料工程師', '資料科學家', '資料分析師', '大數據工程師'],
                '資料庫': ['資料庫工程師', '資料庫管理員'],
                
                # AI/演算法相關
                'AI': ['AI工程師', 'AI研究員', '機器學習工程師', '演算法工程師'],
                
                # 系統相關
                '嵌入式': ['嵌入式系統工程師', '嵌入式軟體工程師'],
                '系統': ['系統分析師', 'Linux系統工程師', '系統架構師'],
                
                # 特殊領域
                '自動化': ['自動化工程師', '自動化測試工程師'],
                '遊戲': ['遊戲開發工程師', '遊戲引擎工程師'],
                '區塊鏈': ['區塊鏈工程師', '區塊鏈開發者'],
                '資安': ['資訊安全工程師', '網路安全工程師'],
                
                # 行動應用
                '行動': ['行動應用開發工程師', 'Android工程師', 'iOS工程師'],
                
                # DevOps/雲端
                '雲端': ['雲端工程師', 'DevOps工程師', '雲端數據工程師'],
                
                # 其他
                'ERP': ['ERP工程師'],
                'IoT': ['IoT工程師'],
                '測試': ['軟體測試工程師', '品質保證工程師']
            }
            
            # 獲取對應的職稱列表
            search_titles = engineer_types.get(job_type, [job_type])
            
            # 使用多個職稱進行搜尋
            relevant_jobs = self.jobs_df[
                self.jobs_df['職稱'].apply(lambda x: any(title in x for title in search_titles)) |
                self.jobs_df['工作技能'].str.contains('|'.join(search_titles), na=False, case=False) |
                self.jobs_df['擅長工具'].str.contains('|'.join(search_titles), na=False, case=False)
            ]
            
            if len(relevant_jobs) == 0:
                print(f"找不到與 {job_type} 相關的職位")
                return []
            
            return relevant_jobs.head(5).to_dict('records')
            
        except Exception as e:
            print(f"搜尋職位時發生錯誤: {str(e)}")
            return []

    def search_relevant_courses(self, skills: List[str]) -> List[Dict]:
        """搜尋相關課程"""
        try:
            relevant_courses = []
            
            # 擴展搜尋關鍵字
            search_keywords = {
                'HTML': ['網頁', 'HTML', 'Web', '前端', '網際網路'],
                'JavaScript': ['JavaScript', 'JS', '程式設計', 'Web', '前端', 'Node.js'],
                'CSS': ['CSS', '網頁設計', 'Web', '前端', '美工'],
                'ReactJS': ['React', '前端框架', 'JavaScript', 'Web'],
                'Python': ['Python', '程式設計', '資料科學'],
                'Java': ['Java', '程式設計', '後端'],
                'C++': ['C++', '程式設計', '系統'],
                'C#': ['C#', '.NET', '程式設計'],
                'SQL': ['SQL', '資料庫', 'Database'],
                'Git': ['Git', '版本控制'],
                'Linux': ['Linux', '作業系統'],
                'AWS': ['AWS', '雲端運算', 'Cloud'],
                'Docker': ['Docker', '容器化', 'Container'],
                '程式設計': ['程式設計', '程式語言', 'Programming', 'Coding'],
                '網頁': ['網頁', 'Web', '前端', '網際網路'],
                '資料庫': ['資料庫', 'Database', 'SQL'],
                '演算法': ['演算法', 'Algorithm', '資料結構'],
                '人工智慧': ['AI', '人工智慧', '機器學習', 'Machine Learning']
            }
            
            # 展開搜尋關鍵字
            expanded_skills = set()
            for skill in skills:
                expanded_skills.add(skill)
                if skill in search_keywords:
                    expanded_skills.update(search_keywords[skill])
            
            # 搜尋課程
            for keyword in expanded_skills:
                courses = self.courses_df[
                    self.courses_df['course_name_zh'].str.contains(keyword, na=False, case=False) |
                    self.courses_df['course_name_en'].str.contains(keyword, na=False, case=False) |
                    self.courses_df['notes'].str.contains(keyword, na=False, case=False)
                ]
                
                if not courses.empty:
                    courses_dict = courses.head(3).apply(lambda row: {
                        '課程名稱': f"{row['course_id']} {row['course_name_zh']}",
                        '課程內容': row['notes'] if pd.notna(row['notes']) else '無說明',
                        '授課教師': row['instructor_zh'],
                        '開課系所': row['department_zh'],
                        '上課時間': f"{row['time']}",
                        '學分數': row['credits']
                    }, axis=1).tolist()
                    
                    relevant_courses.extend(courses_dict)
            
            # 去除重複課程
            seen = set()
            unique_courses = []
            for course in relevant_courses:
                course_id = course['課程名稱'].split()[0]
                if course_id not in seen:
                    seen.add(course_id)
                    unique_courses.append(course)
            
            return unique_courses[:5]  # 限制返回最多5門課程
            
        except Exception as e:
            print(f"搜尋課程時發生錯誤: {str(e)}")
            print("課程資料欄位名稱：", self.courses_df.columns.tolist())
            return []

    def generate_context(self, job_type: str) -> str:
        """生成諮詢上下文"""
        try:
            # 搜尋相關職位
            relevant_jobs = self.search_relevant_jobs(job_type)
            
            # 提取關鍵技能
            skills = set()
            for job in relevant_jobs:
                if isinstance(job.get('擅長工具'), str):
                    skills.update([s.strip() for s in job['擅長工具'].split(',')])
                if isinstance(job.get('工作技能'), str):
                    skills.update([s.strip() for s in job['工作技能'].split(',')])
            
            # 搜尋相關課程
            relevant_courses = self.search_relevant_courses(list(skills))
            
            # 整合資訊
            context = f"""
            職位市場資訊：
            {self._format_jobs(relevant_jobs)}
            
            推薦課程資訊：
            {self._format_courses(relevant_courses)}
            """
            return context
        except Exception as e:
            print(f"生成上下文時發生錯誤: {str(e)}")
            return "無法獲取完整資訊"

    def _format_jobs(self, jobs: List[Dict]) -> str:
        """格式化職位資訊"""
        if not jobs:
            return "暫無相關職位信息"
        
        job_info = []
        for job in jobs:
            info = f"""
            職稱：{job.get('職稱', '未指定')}
            公司名稱：{job.get('公司名稱', '未指定')}
            工作地點：{job.get('工作地點', '未指定')}
            工作經歷要求：{job.get('工作經歷', '未指定')}
            學歷要求：{job.get('學歷要求', '未指定')}
            必備技能：{job.get('擅長工具', '未指定')}
            工作技能：{job.get('工作技能', '未指定')}
            薪資範圍：{job.get('薪資範圍', '未指定')}
            """
            job_info.append(info)
        return "\n---\n".join(job_info)

    def _format_courses(self, courses: List[Dict]) -> str:
        """格式化課程資訊"""
        if not courses:
            return "暫無相關課程信息"
            
        course_info = []
        for course in courses:
            info = f"""
            課程名稱：{course.get('課程名稱', '未指定')}
            課程內容：{course.get('課程內容', '未指定')}
            授課教師：{course.get('授課教師', '未指定')}
            上課時間：{course.get('上課時間', '未指定')}
            課程難度：{course.get('課程難度', '未指定')}
            """
            course_info.append(info)
        return "\n---\n".join(course_info)

    def _format_jobs_summary(self, jobs: List[Dict]) -> str:
        """格式化職缺摘要"""
        if not jobs:
            return "暫無相關職位信息"
        
        summary = "**【職缺摘要】**\n\n"
        for i, job in enumerate(jobs[:5], 1):
            summary += f"{i}. {job.get('公司名稱', '未指定')} - {job.get('職稱', '未指定')}\n"
            summary += f"   - 要求技能：{job.get('擅長工具', '未指定')}\n"
            summary += f"   - 工作技能：{job.get('工作技能', '未指定')}\n"
            summary += f"   - 學歷要求：{job.get('學歷要求', '未指定')}\n"
            summary += f"   - 工作經驗：{job.get('工作經歷', '未指定')}\n\n"
        return summary

    def _format_courses_summary(self, courses: List[Dict]) -> str:
        """格式化課程摘要"""
        if not courses:
            return "暫無相關課程信息"
        
        summary = "**【課程推薦】**\n\n"
        for i, course in enumerate(courses[:5], 1):
            summary += f"{i}. **{course.get('課程名稱', '未指定')}**\n"
            summary += f"   - 授課教師：{course.get('授課教師', '未指定')}\n"
            summary += f"   - 開課系所：{course.get('開課系所', '未指定')}\n"
            summary += f"   - 課程內容：{course.get('課程內容', '未指定')}\n"
            summary += f"   - 上課時間：{course.get('上課時間', '未指定')}\n"
            summary += f"   - 學分數：{course.get('學分數', '未指定')}\n\n"
        return summary

    def generate_career_advice(self, query: str) -> str:
        try:
            # 改進查詢字串處理
            job_type = ''
            if '前端' in query:
                job_type = '前端'
            elif '後端' in query:
                job_type = '後端'
            elif '全端' in query:
                job_type = '全端'
            elif 'AI' in query.upper():
                job_type = 'AI'
            elif '資料' in query:
                job_type = '資料'
            elif '嵌入式' in query:
                job_type = '嵌入式'
            elif '系統' in query:
                job_type = '系統'
            elif '自動化' in query:
                job_type = '自動化'
            elif '遊戲' in query:
                job_type = '遊戲'
            elif '區塊鏈' in query:
                job_type = '區塊鏈'
            elif '資安' in query:
                job_type = '資安'
            elif '行動' in query or 'APP' in query.upper():
                job_type = '行動'
            elif '雲端' in query:
                job_type = '雲端'
            elif 'ERP' in query.upper():
                job_type = 'ERP'
            elif 'IOT' in query.upper():
                job_type = 'IoT'
            elif '測試' in query:
                job_type = '測試'
            else:
                # 如果沒有匹配到特定類型，嘗試提取關鍵字
                keywords = ['工程師', '開發', '程式', '軟體']
                for keyword in keywords:
                    if keyword in query:
                        job_type = query.split(keyword)[0].strip()
                        break
            
            if not job_type:
                return "抱歉，我無法理解您想找什麼類型的工作。請試著提供更具體的工作類型，例如：'前端工程師'、'後端開發'等。"
            
            # 獲取相關職缺和課程
            relevant_jobs = self.search_relevant_jobs(job_type)
            skills = set()
            for job in relevant_jobs:
                if isinstance(job.get('擅長工具'), str):
                    skills.update([s.strip() for s in job['擅長工具'].split(',')])
                if isinstance(job.get('工作技能'), str):
                    skills.update([s.strip() for s in job['工作技能'].split(',')])
            
            relevant_courses = self.search_relevant_courses(list(skills))
            
            # 生成職缺和課程摘要
            jobs_summary = self._format_jobs_summary(relevant_jobs)
            courses_summary = self._format_courses_summary(relevant_courses)
            
            prompt = f"""【職涯顧問建議】

{jobs_summary}

{courses_summary}

根據以上資訊，我為您提供以下建議：

【市場分析】
• 前端工程師目前市場需求強勁
• 大多數職位要求掌握 HTML、CSS、JavaScript 等基礎技能
• ReactJS 是目前最受歡迎的前端框架之一
• 學歷要求普遍為大學以上，但更注重實際技術能力
• 經驗要求彈性，從不拘到 3 年以上都有

【技能建議】
必備技能：
• HTML5/CSS3
• JavaScript/ES6+
• React.js 或其他前端框架
• Git 版本控制
• 響應式網頁設計

加分技能：
• TypeScript
• Node.js
• 前端測試工具
• UI/UX 設計概念
• RESTful API 串接經驗

【學習規劃】
1. 基礎階段：
   • 學習 HTML/CSS 基礎
   • JavaScript 程式設計基礎
   • 網頁排版與響應式設計

2. 進階階段：
   • React.js 框架學習
   • 前端工程化與打包工具
   • API 串接與資料處理

3. 實戰階段：
   • 建立個人作品集
   • 參與開源專案
   • 實作完整網頁專案

【求職建議】
1. 作品準備
   • 建立個人網站展示作品
   • 準備 2-3 個完整專案
   • 維護活躍的 GitHub 帳號

2. 履歷重點
   • 強調專案經驗與技術棧
   • 列出具體的技術成就
   • 附上作品集連結

3. 面試準備
   • 複習前端技術原理
   • 準備常見程式題目
   • 練習線上編程測驗

4. 持續學習
   • 關注前端技術趨勢
   • 參與技術社群活動
   • 訂閱技術部落格或頻道
"""
            
            return prompt
        
        except Exception as e:
            print(f"生成建議時發生錯誤: {str(e)}")
            return "抱歉，在處理您的請求時發生錯誤。請稍後再試。"

def clean_course_data(file_path: str) -> pd.DataFrame:
    """整理課程資料"""
    try:
        # 讀取CSV檔案
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 重新命名欄位，使用更簡潔的名稱
        columns_mapping = {
            'Course #': 'course_id',
            'Point': 'credits',
            '科目名稱': 'course_name_zh',
            'Course Name': 'course_name_en',
            'Instructor': 'instructor',
            'Department and Level': 'department',
            'Session': 'time',
            'Location': 'location',
            'Type of credit': 'course_type',
            'Language': 'language',
            'Note': 'notes'
        }
        
        # 重新命名欄位
        df = df.rename(columns=columns_mapping)
        
        # 清理資料
        # 1. 移除多餘的空白
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()
        
        # 2. 統一空值表示
        df = df.replace('', pd.NA)
        
        # 3. 處理學分數
        if 'credits' in df.columns:
            df['credits'] = pd.to_numeric(df['credits'].str.replace(' ', ''), errors='coerce')
        
        # 4. 將課程類型統一格式
        if 'course_type' in df.columns:
            df['course_type'] = df['course_type'].str.split('/').str[0]
        
        # 5. 將授課語言統一格式
        if 'language' in df.columns:
            df['language'] = df['language'].str.split('/').str[0]
        
        # 6. 建立搜尋用的綜合欄位
        search_cols = ['course_name_zh', 'course_name_en', 'instructor', 'department', 'notes']
        df['search_text'] = df[search_cols].fillna('').agg(' '.join, axis=1)
        
        return df
        
    except Exception as e:
        print(f"整理資料時發生錯誤: {str(e)}")
        return pd.DataFrame()

def main():
    """主程序"""
    try:
        advisor = CareerRAG()
        print("初始化完成！歡迎使用職涯諮詢系統")
        print("請描述你想了解的職位，例如：我想要成為一個前端工程師")
        print("-" * 50)
        
        while True:
            try:
                query = input("\n你的問題（輸入 'quit' 結束）: ").strip()
                
                if query.lower() == 'quit':
                    print("\n「路漫漫其修遠兮，吾將上下而求索」")
                    print("願你在追求理想的道路上砥礪前行！")
                    break
                    
                if not query:
                    print("請告訴我你的職業規劃～")
                    continue
                    
                print("\n正在分析市場數據...", end="")
                for _ in range(3):
                    time.sleep(0.5)
                    print(".", end="", flush=True)
                print("\n")
                
                response = advisor.generate_career_advice(query)
                print("\n【職涯顧問建議】\n")
                print(response)
                print("\n" + "-" * 50)
                
            except Exception as e:
                print(f"處理查詢時發生錯誤: {str(e)}")
                print("請重新輸入您的問題")
                
    except Exception as e:
        print(f"系統初始化失敗: {str(e)}")
        print("請確認所有必要文件都存在且格式正確")

if __name__ == "__main__":
    main()