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
            # 職位類型對應表
            job_type_mapping = {
                '前端': {
                    '職稱': ['前端工程師', '網頁工程師', '.NET工程師', 'Node.js工程師', 'JavaScript工程師', 'Web工程師'],
                    '技能': ['HTML', 'CSS', 'JavaScript', 'React', 'Vue', 'Angular', '網頁開發', 'jQuery', 'TypeScript']
                },
                '後端': {
                    '職稱': ['後端工程師', 'Java工程師', 'Python工程師', '.NET工程師', 'Node.js工程師'],
                    '技能': ['Java', 'Python', 'SQL', 'Node.js', 'API', 'C#', 'PHP']
                },
                '全端': {
                    '職稱': ['全端工程師', '全棧工程師', 'Full Stack工程師'],
                    '技能': ['HTML', 'CSS', 'JavaScript', 'Java', 'Python', 'SQL', 'Node.js']
                }
            }
            
            # 獲取對應的職稱和技能
            job_info = job_type_mapping.get(job_type, {
                '職稱': [job_type],
                '技能': []
            })
            
            # 使用職稱和技能進行搜尋，放寬搜尋條件
            relevant_jobs = self.jobs_df[
                self.jobs_df['職稱'].apply(lambda x: any(title.lower() in str(x).lower() for title in job_info['職稱'])) |
                self.jobs_df['工作技能'].str.contains('|'.join(job_info['技能']), na=False, case=False) |
                self.jobs_df['擅長工具'].str.contains('|'.join(job_info['技能']), na=False, case=False) |
                # 增加更多相關條件
                self.jobs_df['職稱'].str.contains('工程師|開發|程式', na=False, case=False)
            ]
            
            # 計算職位相關性分數
            def calculate_relevance(row):
                score = 0
                # 職稱相關性
                if any(title.lower() in str(row['職稱']).lower() for title in job_info['職稱']):
                    score += 5
                # 技能相關性
                if any(skill.lower() in str(row['擅長工具']).lower() for skill in job_info['技能']):
                    score += 3
                if any(skill.lower() in str(row['工作技能']).lower() for skill in job_info['技能']):
                    score += 3
                return score

            # 添加相關性分數並排序
            relevant_jobs['relevance_score'] = relevant_jobs.apply(calculate_relevance, axis=1)
            relevant_jobs = relevant_jobs.sort_values('relevance_score', ascending=False)
            
            # 返回更多結果（如前100個）
            return relevant_jobs.head(100).to_dict('records')
            
        except Exception as e:
            print(f"搜尋職位時發生錯誤: {str(e)}")
            return []

    def search_relevant_courses(self, skills: List[str]) -> List[Dict]:
        """搜尋相關課程"""
        try:
            print("收到的原始技能列表:", skills)
            
            # 將技能字串分割成單個技能
            individual_skills = set()
            for skill_string in skills:
                parts = skill_string.replace('、', ',').split(',')
                individual_skills.update([s.strip() for s in parts])
            
            print("分割後的個別技能:", individual_skills)
            
            # 定義更精確的課程關鍵字映射
            course_keywords = {
                'JavaScript': ['JavaScript', 'JS', '前端程式', '網頁程式'],
                'HTML': ['HTML', 'Web', '網頁設計', '前端設計'],
                'CSS': ['CSS', '網頁設計', '前端設計'],
                'React': ['React', 'ReactJS', '前端框架'],
                'Angular': ['Angular', 'AngularJS', '前端框架'],
                'Vue': ['Vue', 'VueJS', '前端框架'],
                'Node.js': ['Node', 'NodeJS', '後端程式'],
                'Java': ['Java', 'JAVA', '程式設計'],
                'Python': ['Python', 'python', '程式設計'],
                '程式設計': ['程式設計', '軟體設計', '程式開發'],
                '網頁設計': ['網頁設計', 'Web Design', '前端設計']
            }
            
            # 展開搜尋關鍵字
            search_keywords = set()
            for skill in individual_skills:
                # 檢查技能是否在關鍵字映射中
                if skill in course_keywords:
                    search_keywords.update(course_keywords[skill])
                # 如果不是"不拘"，也將原始技能加入搜尋
                if skill != '不拘':
                    search_keywords.add(skill)
            
            print("最終搜尋關鍵字:", search_keywords)
            
            # 搜尋課程時增加權重計算
            relevant_courses = []
            for keyword in search_keywords:
                mask = (
                    self.courses_df['course_name_zh'].str.contains(keyword, na=False, case=False) |
                    self.courses_df['course_name_en'].str.contains(keyword, na=False, case=False) |
                    self.courses_df['notes'].str.contains(keyword, na=False, case=False)
                )
                matched_courses = self.courses_df[mask]
                
                if not matched_courses.empty:
                    for _, course in matched_courses.iterrows():
                        # 計算相關性分數
                        score = 0
                        if any(k in course['course_name_zh'].lower() for k in ['程式', '設計', '網頁']):
                            score += 3
                        if any(k in course['course_name_en'].lower() for k in ['program', 'web', 'design']):
                            score += 2
                        if any(k in str(course['notes']).lower() for k in ['程式', '設計', '網頁']):
                            score += 1
                        
                        relevant_courses.append({
                            '課程名稱': course['course_name_zh'],
                            '英文名稱': course['course_name_en'],
                            '學分數': course['credits'],
                            '授課教師': course['instructor_zh'],
                            '上課時間': course['time'],
                            '上課地點': course['location'],
                            '課程內容': course['notes'],
                            '相關性分數': score
                        })
            
            # 根據相關性分數排序並去重
            relevant_courses.sort(key=lambda x: x['相關性分數'], reverse=True)
            seen = set()
            filtered_courses = []
            for course in relevant_courses:
                if course['課程名稱'] not in seen:
                    seen.add(course['課程名稱'])
                    filtered_courses.append(course)
            
            return filtered_courses[:30]
            
        except Exception as e:
            print(f"搜尋課程時發生錯誤: {str(e)}")
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
            職位市場資訊：收到的原始技能列表: ['軟體工程系統開發、軟體程式設計', '不拘', 'Github、HTML、JavaScript、CSS、ReactJS', 'MySQL、Oracle、ReactNative', 'HTML、JavaScript、Node.js、AngularJS', '軟體程式設計', '軟體程式設計、網路程式設計', 'Github、Git、AJAX、C#、Java、MS SQL、MySQL、Oracle、PostgreSQL、HTML、JavaScript、jQuery、CSS、ReactJS、AngularJS、VueJS', 'HTML、JavaScript、CSS、ReactJS']
分割後的個別技能: {'軟體程式設計', 'Node.js', 'MS SQL', 'HTML', 'PostgreSQL', 'jQuery', '軟體工程系統開發', 'VueJS', 'Git', '不拘', 'C#', 'MySQL', 'AJAX', 'ReactNative', 'Java', 'AngularJS', 'JavaScript', 'Github', 'Oracle', '網路程式設計', 'ReactJS', 'CSS'}
最終搜尋關鍵字: {'軟體程式設計', 'Node.js', 'MS SQL', 'jQuery', 'HTML', 'PostgreSQL', '軟體工程系統開發', 'VueJS', 'Git', 'C#', 'MySQL', 'AJAX', 'ReactNative', 'Java', 'AngularJS', 'JavaScript', 'Github', 'Oracle', '網路程式設計', 'ReactJS', 'CSS'}
正在搜尋關鍵字: 軟體程式設計
正在搜尋關鍵字: Node.js
正在搜尋關鍵字: MS SQL
正在搜尋關鍵字: jQuery
正在搜尋關鍵字: HTML
正在搜尋關鍵字: PostgreSQL
正在搜尋關鍵字: 軟體工程系統開發
正在搜尋關鍵字: VueJS
正在搜尋關鍵字: Git
找到 66 門相關課程
正在搜尋關鍵字: C#
正在搜尋關鍵字: MySQL
正在搜尋關鍵字: AJAX
正在搜尋關鍵字: ReactNative
正在搜尋關鍵字: Java
找到 2 門相關課程
正在搜尋關鍵字: AngularJS
正在搜尋關鍵字: JavaScript
找到 2 門相關課程
正在搜尋關鍵字: Github
正在搜尋關鍵字: Oracle
正在搜尋關鍵字: 網路程式設計
正在搜尋關鍵字: ReactJS
正在搜尋關鍵字: CSS
去重後共有 39 門課程
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
        for i, job in enumerate(jobs[:50], 1):
            summary += f"{i}. {job.get('職稱', '職稱未指定')}\n"
            
            # 處理其他欄位，使用更友善的預設值
            skills = job.get('擅長工具')
            if not skills or skills == '不拘':
                skills = "依面試能力決定"
            
            work_skills = job.get('工作技能')
            if not work_skills or work_skills == '不拘':
                work_skills = "依面試能力決定"
            
            education = job.get('學歷要求')
            if not education or education == '不拘':
                education = "依實際經驗能力面議"
            
            experience = job.get('工作經歷')
            if not experience or experience == '不拘':
                experience = "依實際經驗能力面議"
            
            summary += f"   - 要求技能：{skills}\n"
            summary += f"   - 工作技能：{work_skills}\n"
            summary += f"   - 學歷要求：{education}\n"
            summary += f"   - 工作經驗：{experience}\n\n"
        
        if len(jobs) > 50:
            summary += f"\n... 還有 {len(jobs) - 50} 個相關職缺 ...\n"
        
        return summary

    def _calculate_course_relevance(self, course: Dict, skills: List[str]) -> int:
        """計算課程相關度分數"""
        score = 0
        course_name = course.get('課程名稱', '').lower()
        course_content = course.get('課程內容', '').lower()
        
        # 關鍵字權重表
        keywords_weight = {
            'javascript': 10,
            'python': 10,
            'java': 9,
            'c++': 9,
            '程式設計': 8,
            '程式': 7,
            'html': 7,
            'css': 7,
            '資料': 6,
            '系統': 5,
            '設計': 4
        }
        
        # 檢查課程名稱中的關鍵字
        for keyword, weight in keywords_weight.items():
            if keyword in course_name:
                score += weight * 2  # 課程名稱中出現的關鍵字權重加倍
                
        # 檢查課程內容中的關鍵字
        for keyword, weight in keywords_weight.items():
            if keyword in course_content:
                score += weight
                
        # 檢查與職缺要求技能的匹配度
        for skill in skills:
            if skill.lower() in course_name or skill.lower() in course_content:
                score += 15  # 與職缺技能相關的課程獲得額外分數
        
        return score

    def _format_courses_summary(self, courses: List[Dict], skills: List[str]) -> str:
        """格式化課程摘要"""
        if not courses:
            return "暫無相關課程信息"
        
        # 為每個課程計算相關度分數
        scored_courses = [
            (course, self._calculate_course_relevance(course, skills))
            for course in courses
        ]
        
        # 根據分數排序課程
        sorted_courses = sorted(scored_courses, key=lambda x: x[1], reverse=True)
        
        summary = "**【課程推薦】**\n\n"
        for i, (course, score) in enumerate(sorted_courses[:30], 1):
            summary += f"{i}. **{course.get('課程名稱', '未指定')}**\n"
            summary += f"   - 英文名稱：{course.get('英文名稱', '未指定')}\n"
            summary += f"   - 授課教師：{course.get('授課教師', '未指定')}\n"
            summary += f"   - 開課系所：{course.get('開課系所', '未指定')}\n"
            summary += f"   - 課程內容：{course.get('課程內容', '未指定')}\n"
            summary += f"   - 上課時間：{course.get('上課時間', '未指定')}\n"
            summary += f"   - 學分數：{course.get('學分數', '未指定')}\n\n"
        
        if len(courses) > 30:
            summary += f"\n... 還有 {len(courses) - 30} 門相關課程 ...\n"
        
        return summary

    def generate_career_advice(self, query: str) -> str:
        try:
            # 獲取職缺和課程資訊
            relevant_jobs = self.search_relevant_jobs(query)
            skills = set()
            for job in relevant_jobs:
                if isinstance(job.get('擅長工具'), str):
                    skills.update([s.strip() for s in job['擅長工具'].split(',')])
                if isinstance(job.get('工作技能'), str):
                    skills.update([s.strip() for s in job['工作技能'].split(',')])
            
            relevant_courses = self.search_relevant_courses(list(skills))
            
            # 格式化職缺和課程摘要
            jobs_summary = self._format_jobs_summary(relevant_jobs)
            courses_summary = self._format_courses_summary(relevant_courses, list(skills))
            
            # 準備上下文資訊
            prompt = f"""
            你現在是一位充滿智慧的大學教授，正在為學生提供職涯諮詢。
            這位學生對{query}感興趣。
            請以關心且睿智的語氣，運用文言文風格，為這位學生提供建議。
            """
            
            # 使用 Ollama 生成建議
            response = self.client.generate(model='llama3.2', prompt=prompt)
            advice = response['response']
            
            # 組合完整回應
            return f"{jobs_summary}\n\n{courses_summary}\n\n{advice}"
            
        except Exception as e:
            print(f"生成建議時發生錯誤: {str(e)}")
            return "無法生成完整建議"

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