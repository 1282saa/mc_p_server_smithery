import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
import os

# 엑셀 파일 경로
excel_file = 'smithery_mcp_server.xlsx'

# 모든 시트 데이터 로드
print("엑셀 파일에서 데이터 로드 중...")
featured_category = pd.read_excel(excel_file, sheet_name="featured_category", skiprows=3,
                        usecols=lambda x: not x.startswith('Unnamed: 0'))
web_search = pd.read_excel(excel_file, sheet_name="web search", skiprows=3,
                        usecols=lambda x: not x.startswith('Unnamed: 0'))
browser_automation = pd.read_excel(excel_file, sheet_name="browser automation", skiprows=3,
                        usecols=lambda x: not x.startswith('Unnamed: 0'))
memory_management = pd.read_excel(excel_file, sheet_name="memory management", skiprows=3,
                        usecols=lambda x: not x.startswith('Unnamed: 0'))
dynamic_web_development = pd.read_excel(excel_file, sheet_name="Dynamic Web Development", skiprows=3,
                        usecols=lambda x: not x.startswith('Unnamed: 0'))
application_integration_tools = pd.read_excel(excel_file, sheet_name="Application Integration Tools", skiprows=3,
                        usecols=lambda x: not x.startswith('Unnamed: 0'))
ai_integration_solutions = pd.read_excel(excel_file, sheet_name="AI Integration Solutions", skiprows=3,
                        usecols=lambda x: not x.startswith('Unnamed: 0'))
financial_data_analysis = pd.read_excel(excel_file, sheet_name="Financial Data & Analysis", skiprows=3,
                        usecols=lambda x: not x.startswith('Unnamed: 0'))

# 시트별 데이터 정보 출력
def print_sheet_info(name, df):
    print(f"\n=== {name} 시트 정보 ===")
    print(f"행 수: {len(df)}")
    print(f"열 이름: {', '.join(df.columns)}")
    if len(df) > 0:
        print("\n첫 3개 행:")
        print(df.head(3))

# 각 시트 정보 출력
print_sheet_info("featured_category", featured_category)
print_sheet_info("web_search", web_search)
print_sheet_info("browser_automation", browser_automation)
print_sheet_info("memory_management", memory_management)
print_sheet_info("dynamic_web_development", dynamic_web_development)
print_sheet_info("application_integration_tools", application_integration_tools)
print_sheet_info("ai_integration_solutions", ai_integration_solutions)
print_sheet_info("financial_data_analysis", financial_data_analysis)

# 텍스트 기반 분류 함수 (TF-IDF와 K-means 사용)
def classify_by_text(df, text_column, num_clusters=5, random_state=42):
    """텍스트 데이터를 기반으로 클러스터링하는 함수"""
    # 결측치 처리
    df = df.dropna(subset=[text_column])
    
    if len(df) < num_clusters:
        print(f"데이터 개수({len(df)})가 클러스터 수({num_clusters})보다 적습니다. 클러스터 수를 조정합니다.")
        num_clusters = max(2, len(df) // 2)
    
    # TF-IDF 벡터화
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column].astype(str))
    
    # K-means 클러스터링 수행
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    df['cluster'] = kmeans.fit_predict(tfidf_matrix)
    
    # 클러스터별 주요 키워드 추출
    feature_names = tfidf_vectorizer.get_feature_names_out()
    cluster_keywords = {}
    
    # 각 클러스터의 중심점에서 가장 중요한 단어 추출
    for i in range(num_clusters):
        # 클러스터 중심점의 단어 중요도
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        # 각 클러스터의 상위 5개 키워드 추출
        keywords = [feature_names[idx] for idx in order_centroids[i, :5]]
        cluster_keywords[i] = keywords
    
    return df, cluster_keywords

# 각 시트별로 분류 작업 수행
def process_sheet(sheet_name, df, text_column='description', num_clusters=5):
    """각 시트별로 분류 작업을 수행하고 결과를 저장하는 함수"""
    print(f"\n\n=== {sheet_name} 시트 분류 시작 ===")
    
    if len(df) == 0:
        print(f"{sheet_name} 시트에 데이터가 없습니다.")
        return None
    
    # 필요한 컬럼이 있는지 확인
    if text_column not in df.columns:
        print(f"{sheet_name} 시트에 {text_column} 컬럼이 없습니다.")
        print(f"사용 가능한 컬럼: {', '.join(df.columns)}")
        # 대체 텍스트 컬럼 찾기
        text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                       ['description', 'name', 'text', 'title', 'summary'])]
        if text_columns:
            text_column = text_columns[0]
            print(f"대체 텍스트 컬럼으로 {text_column}을(를) 사용합니다.")
        else:
            print(f"{sheet_name} 시트에 적절한 텍스트 컬럼을 찾을 수 없습니다.")
            return None
    
    # 분류 작업 수행
    try:
        classified_df, keywords = classify_by_text(df, text_column, num_clusters)
        
        # 결과 요약
        print(f"\n{sheet_name} 클러스터 결과:")
        cluster_counts = Counter(classified_df['cluster'])
        for cluster, count in sorted(cluster_counts.items()):
            print(f"클러스터 {cluster}: {count}개 항목, 키워드: {', '.join(keywords[cluster])}")
        
        # CSV 파일로 저장
        output_file = f"{sheet_name.replace(' ', '_').lower()}_classified.csv"
        classified_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"{output_file} 파일로 저장되었습니다.")
        
        return classified_df
    
    except Exception as e:
        print(f"{sheet_name} 시트 처리 중 오류 발생: {str(e)}")
        return None

# 시트별 분류 수행
sheets_data = {
    "featured_category": featured_category,
    "web_search": web_search,
    "browser_automation": browser_automation,
    "memory_management": memory_management,
    "dynamic_web_development": dynamic_web_development,
    "application_integration_tools": application_integration_tools,
    "ai_integration_solutions": ai_integration_solutions,
    "financial_data_analysis": financial_data_analysis
}

# 결과 저장할 디렉토리 생성
output_dir = "classified_results"
os.makedirs(output_dir, exist_ok=True)

# 각 시트별 분류 수행
results = {}
for sheet_name, df in sheets_data.items():
    # 각 시트의 데이터 크기에 따라 클러스터 수 조정
    num_clusters = min(5, max(2, len(df) // 20)) if len(df) > 0 else 2
    result_df = process_sheet(sheet_name, df, num_clusters=num_clusters)
    if result_df is not None:
        results[sheet_name] = result_df

print("\n\n=== 분류 작업 완료 ===")
print(f"총 {len(results)}개 시트에 대한 분류가 완료되었습니다.") 