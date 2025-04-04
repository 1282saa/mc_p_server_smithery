import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

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

# 결과 저장할 디렉토리 생성
output_dir = "topic_modeling_results"
os.makedirs(output_dir, exist_ok=True)

# LDA 토픽 모델링을 이용한 분류 함수
def extract_topics_lda(df, text_column, num_topics=5, num_words=10):
    """LDA 토픽 모델링을 이용해 문서의 주제를 추출하는 함수"""
    # 결측치 처리
    df = df.dropna(subset=[text_column])
    texts = df[text_column].astype(str).tolist()
    
    # 문서가 너무 적으면 주제 수 조정
    if len(texts) < num_topics:
        print(f"데이터 개수({len(texts)})가 주제 수({num_topics})보다 적습니다. 주제 수를 조정합니다.")
        num_topics = max(2, len(texts) // 2)
    
    # 문서-단어 행렬 생성
    cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = cv.fit_transform(texts)
    
    # LDA 모델 학습
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)
    
    # 문서별 토픽 할당
    topic_results = lda.transform(dtm)
    df['topic'] = topic_results.argmax(axis=1)
    
    # 토픽별 주요 단어 추출
    feature_names = cv.get_feature_names_out()
    topics_keywords = {}
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-num_words-1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics_keywords[topic_idx] = top_words
    
    return df, topics_keywords, cv, lda

# 워드클라우드 생성 함수
def generate_wordcloud(topics_keywords, topic_idx, output_dir, sheet_name):
    """토픽별 워드클라우드를 생성하는 함수"""
    keywords = topics_keywords[topic_idx]
    word_freq = {word: (len(keywords) - i) * 10 for i, word in enumerate(keywords)}
    
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          colormap='viridis', max_words=50)
    wordcloud.generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{sheet_name} - 토픽 {topic_idx} 워드클라우드')
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"{sheet_name.replace(' ', '_').lower()}_topic_{topic_idx}_wordcloud.png")
    plt.savefig(filename)
    plt.close()
    return filename

# 토픽별 문서 분포 시각화 함수
def plot_topic_distribution(df, sheet_name, output_dir):
    """토픽별 문서 분포를 시각화하는 함수"""
    topic_counts = df['topic'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=topic_counts.index, y=topic_counts.values)
    plt.xlabel('토픽 번호')
    plt.ylabel('문서 수')
    plt.title(f'{sheet_name} 토픽별 문서 분포')
    plt.xticks(range(len(topic_counts)))
    
    filename = os.path.join(output_dir, f"{sheet_name.replace(' ', '_').lower()}_topic_distribution.png")
    plt.savefig(filename)
    plt.close()
    return filename

# 각 시트별로 토픽 모델링 수행
def process_sheet_topics(sheet_name, df, text_column='description', num_topics=5):
    """각 시트별로 토픽 모델링을 수행하고 결과를 저장하는 함수"""
    print(f"\n\n=== {sheet_name} 시트 토픽 모델링 시작 ===")
    
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
    
    # 토픽 모델링 수행
    try:
        df_topics, topics_keywords, cv, lda = extract_topics_lda(df, text_column, num_topics)
        
        # 결과 요약
        print(f"\n{sheet_name} 토픽 모델링 결과:")
        for topic_idx, keywords in topics_keywords.items():
            print(f"토픽 {topic_idx}: {', '.join(keywords)}")
        
        # 토픽별 문서 수 계산
        topic_counts = df_topics['topic'].value_counts().sort_index()
        for topic_idx, count in topic_counts.items():
            print(f"토픽 {topic_idx}: {count}개 문서")
        
        # 워드클라우드 생성
        for topic_idx in topics_keywords.keys():
            wordcloud_file = generate_wordcloud(topics_keywords, topic_idx, output_dir, sheet_name)
            print(f"토픽 {topic_idx} 워드클라우드 저장됨: {wordcloud_file}")
        
        # 토픽 분포 시각화
        dist_file = plot_topic_distribution(df_topics, sheet_name, output_dir)
        print(f"토픽 분포 시각화 저장됨: {dist_file}")
        
        # CSV 파일로 저장
        output_file = os.path.join(output_dir, f"{sheet_name.replace(' ', '_').lower()}_topics.csv")
        df_topics.to_csv(output_file, index=False, encoding='utf-8')
        print(f"토픽 모델링 결과 저장됨: {output_file}")
        
        return df_topics, topics_keywords
    
    except Exception as e:
        print(f"{sheet_name} 시트 토픽 모델링 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
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

# 각 시트별 토픽 모델링 수행
results = {}
for sheet_name, df in sheets_data.items():
    # 시트의 데이터 크기에 따라 토픽 수 조정
    num_topics = min(5, max(2, len(df) // 20)) if len(df) > 0 else 2
    result = process_sheet_topics(sheet_name, df, num_topics=num_topics)
    if result is not None:
        results[sheet_name] = result

print("\n\n=== 토픽 모델링 완료 ===")
print(f"총 {len(results)}개 시트에 대한 토픽 모델링이 완료되었습니다.")

# 전체 결과 요약 보고서 작성
with open(os.path.join(output_dir, "토픽_모델링_결과_요약.txt"), "w", encoding="utf-8") as f:
    f.write("# 토픽 모델링 결과 요약\n\n")
    for sheet_name, (df_topics, topics_keywords) in results.items():
        f.write(f"\n## {sheet_name} 시트\n")
        f.write(f"- 총 문서 수: {len(df_topics)}\n")
        f.write(f"- 토픽 수: {len(topics_keywords)}\n\n")
        
        # 토픽별 키워드와 문서 수
        topic_counts = df_topics['topic'].value_counts().sort_index()
        f.write("### 토픽별 키워드 및 문서 수\n")
        for topic_idx, keywords in topics_keywords.items():
            count = topic_counts.get(topic_idx, 0)
            f.write(f"- 토픽 {topic_idx} ({count}개 문서): {', '.join(keywords)}\n")
        f.write("\n")

print(f"토픽 모델링 결과 요약 보고서가 생성되었습니다: {os.path.join(output_dir, '토픽_모델링_결과_요약.txt')}") 