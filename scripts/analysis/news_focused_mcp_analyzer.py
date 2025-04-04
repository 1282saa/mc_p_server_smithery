import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

# 결과 저장 디렉토리 생성
output_dir = "news_mcp_results"
os.makedirs(output_dir, exist_ok=True)

# 엑셀 파일 경로 수정
excel_file = 'smithery/smithery_mcp_server.xlsx'

print("엑셀 파일에서 데이터 로드 중...")
# 모든 시트 데이터 로드
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

# 모든 데이터프레임을 하나로 합치기
all_sheets = {
    "featured_category": featured_category,
    "web_search": web_search, 
    "browser_automation": browser_automation,
    "memory_management": memory_management,
    "dynamic_web_development": dynamic_web_development,
    "application_integration_tools": application_integration_tools,
    "ai_integration_solutions": ai_integration_solutions,
    "financial_data_analysis": financial_data_analysis
}

# 모든 데이터프레임에 시트 이름 컬럼 추가하고 하나로 합치기
dfs = []
for sheet_name, df in all_sheets.items():
    if len(df) > 0:
        df = df.copy()
        df['sheet_name'] = sheet_name
        dfs.append(df)

# 전체 데이터프레임 만들기
all_data = pd.concat(dfs, ignore_index=True)
print(f"총 {len(all_data)}개의 MCP 서버 데이터 로드 완료")

# 기자/신문사 업무 관련 키워드 정의
news_keywords = [
    # 취재 관련
    "news", "journalism", "reporter", "interview", "research", "investigation", "story", "article", 
    "media", "press", "newspaper", "broadcast", "publishing", "editorial", "editor", "뉴스", "기자", 
    "취재", "인터뷰", "조사", "기사", "미디어", "언론", "신문", "방송", "출판", "편집",
    
    # 데이터 분석 관련
    "data analysis", "statistics", "trends", "insights", "visualization", "charts", "graphs",
    "데이터 분석", "통계", "트렌드", "인사이트", "시각화", "차트", "그래프",
    
    # 콘텐츠 생성 관련
    "content", "writing", "summarize", "summarization", "paraphrase", "translation", 
    "grammar", "proofreading", "edit", "headline", "title", "caption",
    "콘텐츠", "글쓰기", "요약", "번역", "문법", "교정", "편집", "헤드라인", "제목", "캡션",
    
    # 팩트체크 관련
    "fact check", "verification", "accuracy", "source", "reference", "authenticate", 
    "validate", "credibility", "팩트체크", "검증", "정확성", "출처", "참조", "신뢰성",
    
    # 소셜미디어 관련
    "social media", "twitter", "facebook", "instagram", "social network", "trending", 
    "viral", "소셜미디어", "트위터", "페이스북", "인스타그램", "소셜 네트워크", "트렌딩", "바이럴",
    
    # AI 관련
    "ai writing", "ai journalism", "automated journalism", "robot journalism", 
    "ai 글쓰기", "ai 저널리즘", "자동화 저널리즘", "로봇 저널리즘",
    
    # 웹 크롤링/스크래핑 관련
    "web crawling", "scraping", "data extraction", "web automation",
    "웹 크롤링", "스크래핑", "데이터 추출", "웹 자동화"
]

# 기자 업무 시나리오 정의
journalism_scenarios = {
    "기사 작성 보조": "AI를 활용한 기사 초안 작성, 문법 검사, 제목 생성, 요약 등의 기능이 필요합니다.",
    "팩트 체크 및 검증": "정보의 정확성 검증, 출처 확인, 인용구 검색 등의 기능이 필요합니다.",
    "데이터 저널리즘": "데이터 분석, 시각화, 통계 처리, 그래프 생성 등의 기능이 필요합니다.",
    "트렌드 분석": "소셜 미디어 트렌드, 뉴스 트렌드, 키워드 분석 등의 기능이 필요합니다.",
    "멀티미디어 콘텐츠": "이미지 처리, 오디오 변환, 비디오 편집 등의 기능이 필요합니다.",
    "취재 보조": "인터뷰 정리, 녹취록 생성, 질문 생성, 배경 조사 등의 기능이 필요합니다.",
    "국제 뉴스 보도": "번역, 다국어 처리, 국제 뉴스 모니터링 등의 기능이 필요합니다.",
    "아카이브 및 검색": "과거 기사 검색, 정보 아카이브, 관련 기사 찾기 등의 기능이 필요합니다."
}

# 신문사 요구사항에 맞는 MCP 서버 찾기 함수
def find_news_related_mcps(df, keywords, top_n=20, description_col='설명'):
    """
    주어진 키워드를 기반으로 신문사 관련 MCP 서버를 찾는 함수
    
    Args:
        df: 데이터프레임
        keywords: 검색할 키워드 목록
        top_n: 반환할 상위 결과 수
        description_col: 설명이 있는 열 이름
    
    Returns:
        관련도가 높은 MCP 서버 목록 데이터프레임
    """
    # 설명 열이 비어있지 않은 행만 필터링
    df = df.dropna(subset=[description_col])
    
    # 각 설명에서 키워드 매칭 점수 계산
    scores = []
    for idx, row in df.iterrows():
        description = row[description_col].lower()
        score = sum(1 for keyword in keywords if keyword.lower() in description)
        scores.append(score)
    
    # 점수 열 추가
    df_scored = df.copy()
    df_scored['relevance_score'] = scores
    
    # 점수 기준 내림차순 정렬 후 상위 N개 반환
    return df_scored.sort_values('relevance_score', ascending=False).head(top_n)

# 시나리오별 MCP 서버 추천 함수
def recommend_mcps_for_scenario(df, scenario, scenario_description, top_n=10):
    """
    특정 저널리즘 시나리오에 맞는 MCP 서버를 추천하는 함수
    
    Args:
        df: 데이터프레임
        scenario: 시나리오 이름
        scenario_description: 시나리오 설명
        top_n: 추천할 서버 수
    
    Returns:
        추천 MCP 서버 목록
    """
    # TF-IDF 벡터화
    tfidf = TfidfVectorizer(max_features=5000)
    
    # 설명 열이 비어 있지 않은 행만 사용
    valid_df = df.dropna(subset=['설명'])
    
    # TF-IDF 행렬 생성
    tfidf_matrix = tfidf.fit_transform(valid_df['설명'].astype(str))
    
    # 시나리오 설명을 쿼리로 변환
    query_vec = tfidf.transform([scenario_description])
    
    # 코사인 유사도 계산
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # 유사도 점수 추가
    results_df = valid_df.copy()
    results_df['similarity_score'] = cosine_similarities
    
    # 유사도 기준 내림차순 정렬 후 상위 N개 반환
    return results_df.sort_values('similarity_score', ascending=False).head(top_n)

# 사용 횟수와 관련도를 고려한 종합 점수 계산 함수
def calculate_composite_score(df):
    """
    사용 횟수와 관련도를 모두 고려한 종합 점수를 계산하는 함수
    
    Args:
        df: 관련도 점수가 포함된 데이터프레임
    
    Returns:
        종합 점수가 추가된 데이터프레임
    """
    # usage_count 열이 string 타입인 경우 숫자로 변환
    if 'usage_count' in df.columns and df['usage_count'].dtype == 'object':
        # 'k' 접미사 처리 (예: 94.24k -> 94240)
        df['usage_count_numeric'] = df['usage_count'].apply(
            lambda x: float(x.replace('k', '')) * 1000 if isinstance(x, str) and 'k' in x 
            else float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() 
            else 0
        )
    else:
        df['usage_count_numeric'] = 0
    
    # 관련도 점수와 사용 횟수를 정규화 (Min-Max 스케일링)
    if 'relevance_score' in df.columns:
        max_relevance = df['relevance_score'].max()
        min_relevance = df['relevance_score'].min()
        if max_relevance > min_relevance:
            df['relevance_normalized'] = (df['relevance_score'] - min_relevance) / (max_relevance - min_relevance)
        else:
            df['relevance_normalized'] = 1.0
    
    if 'similarity_score' in df.columns:
        max_similarity = df['similarity_score'].max()
        min_similarity = df['similarity_score'].min()
        if max_similarity > min_similarity:
            df['similarity_normalized'] = (df['similarity_score'] - min_similarity) / (max_similarity - min_similarity)
        else:
            df['similarity_normalized'] = 1.0
    
    max_usage = df['usage_count_numeric'].max()
    min_usage = df['usage_count_numeric'].min()
    if max_usage > min_usage:
        df['usage_normalized'] = (df['usage_count_numeric'] - min_usage) / (max_usage - min_usage)
    else:
        df['usage_normalized'] = 1.0
    
    # 종합 점수 계산 (관련도 70%, 사용 횟수 30%)
    if 'relevance_normalized' in df.columns:
        df['composite_score'] = 0.7 * df['relevance_normalized'] + 0.3 * df['usage_normalized']
    elif 'similarity_normalized' in df.columns:
        df['composite_score'] = 0.7 * df['similarity_normalized'] + 0.3 * df['usage_normalized']
    
    return df

# 메인 분석 프로세스
print("\n1. 기자/신문사 관련 MCP 서버 탐색 중...")
news_related_mcps = find_news_related_mcps(all_data, news_keywords, top_n=50)
news_related_mcps = calculate_composite_score(news_related_mcps)
news_related_mcps_sorted = news_related_mcps.sort_values('composite_score', ascending=False)

# 상위 결과 저장
news_related_mcps_sorted.to_csv(os.path.join(output_dir, "신문사_관련_MCP서버_TOP50.csv"), encoding='utf-8-sig', index=False)
print(f"기자/신문사 관련 상위 50개 MCP 서버 목록 저장 완료: {os.path.join(output_dir, '신문사_관련_MCP서버_TOP50.csv')}")

# 시트별 분포 시각화
sheet_counts = news_related_mcps_sorted.head(50)['sheet_name'].value_counts()
plt.figure(figsize=(12, 6))
sns.barplot(x=sheet_counts.index, y=sheet_counts.values)
plt.title('신문사 관련 상위 50개 MCP 서버의 시트별 분포')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "신문사_관련_MCP_시트별분포.png"))
plt.close()

# 각 시나리오별 추천 MCP 서버 찾기
print("\n2. 기자 업무 시나리오별 MCP 서버 추천 중...")
scenario_results = {}

for scenario, description in journalism_scenarios.items():
    print(f"  - '{scenario}' 시나리오에 적합한 MCP 서버 분석 중...")
    recommended_mcps = recommend_mcps_for_scenario(all_data, scenario, description, top_n=10)
    recommended_mcps = calculate_composite_score(recommended_mcps)
    recommended_mcps_sorted = recommended_mcps.sort_values('composite_score', ascending=False)
    
    # 결과 저장
    scenario_file = f"{scenario.replace(' ', '_')}_추천_MCP서버.csv"
    recommended_mcps_sorted.to_csv(os.path.join(output_dir, scenario_file), encoding='utf-8-sig', index=False)
    scenario_results[scenario] = recommended_mcps_sorted
    
    print(f"    → {scenario} 시나리오 분석 완료 (상위 10개 서버 저장)")

# 시나리오별 결과를 하나의 HTML 파일로 통합
html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>기자 업무 시나리오별 MCP 서버 추천 결과</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        h2 { color: #0066cc; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin-top: 10px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #eef7ff; }
        .description { font-style: italic; color: #666; margin-bottom: 15px; }
    </style>
</head>
<body>
    <h1>기자 업무 시나리오별 MCP 서버 추천 결과</h1>
    <p>각 시나리오에 맞는 상위 10개의 MCP 서버를 추천합니다.</p>
"""

for scenario, description in journalism_scenarios.items():
    html_content += f"""
    <h2>{scenario}</h2>
    <p class="description">{description}</p>
    <table>
        <tr>
            <th>서버 이름</th>
            <th>설명</th>
            <th>URL</th>
            <th>유사도 점수</th>
            <th>사용 횟수</th>
            <th>종합 점수</th>
        </tr>
    """
    
    for _, row in scenario_results[scenario].iterrows():
        html_content += f"""
        <tr>
            <td>{row.get('name', '')}</td>
            <td>{row.get('설명', '')}</td>
            <td><a href="{row.get('url', '')}" target="_blank">{row.get('url', '')}</a></td>
            <td>{row.get('similarity_score', 0):.4f}</td>
            <td>{row.get('usage_count', '')}</td>
            <td>{row.get('composite_score', 0):.4f}</td>
        </tr>
        """
    
    html_content += """
    </table>
    """

html_content += """
</body>
</html>
"""

with open(os.path.join(output_dir, "기자_업무_시나리오별_MCP서버_추천.html"), "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"\n시나리오별 추천 결과 HTML 파일 생성 완료: {os.path.join(output_dir, '기자_업무_시나리오별_MCP서버_추천.html')}")

# 종합 분석 결과 마크다운 파일 생성
with open(os.path.join(output_dir, "신문사_MCP서버_분석_종합보고서.md"), "w", encoding="utf-8") as f:
    f.write("# 신문사 MCP 서버 분석 종합 보고서\n\n")
    
    f.write("## 1. 개요\n\n")
    f.write("이 보고서는 기자와 신문사 업무에 활용할 수 있는 MCP(Model Context Protocol) 서버를 분석한 결과입니다. ")
    f.write(f"총 {len(all_data)}개의 MCP 서버를 분석하여 기자 업무와 관련성이 높은 서버를 추천합니다.\n\n")
    
    f.write("## 2. 주요 발견 사항\n\n")
    
    f.write("### 2.1 기자 업무 관련 상위 MCP 서버\n\n")
    f.write("신문사 관련 키워드를 기반으로 찾은, 관련성이 높은 상위 10개 MCP 서버는 다음과 같습니다:\n\n")
    
    f.write("| 서버 이름 | 설명 | 사용 횟수 | 관련도 점수 |\n")
    f.write("|---------|------|-----------|------------|\n")
    for _, row in news_related_mcps_sorted.head(10).iterrows():
        f.write(f"| {row.get('name', '')} | {row.get('설명', '')[:100]}... | {row.get('usage_count', '')} | {row.get('relevance_score', 0)} |\n")
    
    f.write("\n### 2.2 시트별 관련 MCP 서버 분포\n\n")
    f.write("신문사 관련 상위 50개 MCP 서버의 시트별 분포는 다음과 같습니다:\n\n")
    
    for sheet, count in sheet_counts.items():
        f.write(f"- {sheet}: {count}개\n")
    
    f.write("\n## 3. 시나리오별 추천 MCP 서버\n\n")
    
    for scenario, description in journalism_scenarios.items():
        f.write(f"### 3.{list(journalism_scenarios.keys()).index(scenario) + 1} {scenario}\n\n")
        f.write(f"**{description}**\n\n")
        f.write("추천 MCP 서버:\n\n")
        
        f.write("| 서버 이름 | 설명 | 유사도 점수 | 사용 횟수 |\n")
        f.write("|---------|------|-----------|------------|\n")
        for _, row in scenario_results[scenario].head(5).iterrows():
            f.write(f"| {row.get('name', '')} | {row.get('설명', '')[:100]}... | {row.get('similarity_score', 0):.4f} | {row.get('usage_count', '')} |\n")
        
        f.write("\n")
    
    f.write("## 4. 결론 및 권장사항\n\n")
    f.write("분석 결과, 기자 업무와 관련하여 다음과 같은 MCP 서버 유형이 가장 유용할 것으로 판단됩니다:\n\n")
    
    top_sheets = sheet_counts.head(3).index.tolist()
    f.write(f"1. **{top_sheets[0]}** 카테고리의 서버들: 이 카테고리에는 기자 업무와 관련성이 높은 서버가 가장 많이 포함되어 있습니다.\n")
    if len(top_sheets) > 1:
        f.write(f"2. **{top_sheets[1]}** 카테고리의 서버들: 두 번째로 관련성이 높은 카테고리입니다.\n")
    if len(top_sheets) > 2:
        f.write(f"3. **{top_sheets[2]}** 카테고리의 서버들: 세 번째로 관련성이 높은 카테고리입니다.\n")
    
    f.write("\n신문사 업무에 MCP 서버를 도입할 때는 다음 사항을 고려하세요:\n\n")
    f.write("- 서버의 사용 횟수(usage_count)는 인기도와 안정성의 간접적인 지표가 될 수 있습니다.\n")
    f.write("- 각 시나리오별 추천 서버를 검토하여 실제 업무 흐름에 맞는 서버를 선택하세요.\n")
    f.write("- 한국어 지원 여부를 확인하고, 필요시 번역 서버와 함께 활용하는 방안도 고려하세요.\n")
    f.write("- 중요한 보안 정보를 다룰 때는 Local 타입의 서버를 우선적으로 고려하세요.\n")

print(f"\n종합 분석 보고서 생성 완료: {os.path.join(output_dir, '신문사_MCP서버_분석_종합보고서.md')}")
print("\n분석이 완료되었습니다. 신문사 업무에 적합한 MCP 서버 추천 결과를 확인하세요.") 