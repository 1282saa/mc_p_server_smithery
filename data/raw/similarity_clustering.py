import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import umap
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import warnings
warnings.filterwarnings('ignore')

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
output_dir = "similarity_clustering_results"
os.makedirs(output_dir, exist_ok=True)

# TF-IDF 벡터화 및 유사도 계산 함수
def compute_similarity_matrix(df, text_column):
    """TF-IDF 벡터화를 통해 문서 간 유사도 행렬을 계산하는 함수"""
    # 결측치 처리
    df = df.dropna(subset=[text_column])
    texts = df[text_column].astype(str).tolist()
    
    # TF-IDF 벡터화
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    
    # 코사인 유사도 계산
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    return df, similarity_matrix, tfidf_matrix, tfidf_vectorizer

# 계층적 클러스터링 수행 함수
def hierarchical_clustering(similarity_matrix, num_clusters=5, method='average'):
    """유사도 행렬을 기반으로 계층적 클러스터링을 수행하는 함수"""
    # 유사도를 거리로 변환 (1 - 유사도)
    distance_matrix = 1 - similarity_matrix
    
    # 계층적 클러스터링 수행
    Z = linkage(distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)], method=method)
    
    # 클러스터 할당
    clusters = fcluster(Z, num_clusters, criterion='maxclust')
    
    return clusters, Z

# 차원 축소 및 시각화 함수
def visualize_clusters(tfidf_matrix, clusters, sheet_name, output_dir, method='tsne'):
    """TF-IDF 행렬을 2차원으로 축소하여 클러스터를 시각화하는 함수"""
    # 차원 축소
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        embedding = reducer.fit_transform(tfidf_matrix.toarray())
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(tfidf_matrix.toarray())
    
    # 시각화
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='클러스터')
    plt.title(f'{sheet_name} 클러스터 시각화 ({method.upper()})')
    plt.tight_layout()
    
    # 파일 저장
    filename = os.path.join(output_dir, f"{sheet_name.replace(' ', '_').lower()}_clusters_{method}.png")
    plt.savefig(filename)
    plt.close()
    return filename

# 덴드로그램 시각화 함수
def visualize_dendrogram(Z, sheet_name, output_dir):
    """계층적 클러스터링 결과를 덴드로그램으로 시각화하는 함수"""
    plt.figure(figsize=(14, 8))
    plt.title(f'{sheet_name} 계층적 클러스터링 덴드로그램')
    dendrogram(Z, leaf_rotation=90, leaf_font_size=8, truncate_mode='lastp', p=30)
    plt.tight_layout()
    
    # 파일 저장
    filename = os.path.join(output_dir, f"{sheet_name.replace(' ', '_').lower()}_dendrogram.png")
    plt.savefig(filename)
    plt.close()
    return filename

# 유사도 히트맵 시각화 함수
def visualize_similarity_heatmap(similarity_matrix, sheet_name, output_dir, max_size=100):
    """문서 간 유사도를 히트맵으로 시각화하는 함수"""
    # 너무 큰 행렬은 샘플링
    if similarity_matrix.shape[0] > max_size:
        indices = np.random.choice(similarity_matrix.shape[0], max_size, replace=False)
        sim_matrix_sample = similarity_matrix[np.ix_(indices, indices)]
    else:
        sim_matrix_sample = similarity_matrix
    
    # 히트맵 그리기
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_matrix_sample, cmap='viridis')
    plt.title(f'{sheet_name} 문서 유사도 히트맵')
    plt.tight_layout()
    
    # 파일 저장
    filename = os.path.join(output_dir, f"{sheet_name.replace(' ', '_').lower()}_similarity_heatmap.png")
    plt.savefig(filename)
    plt.close()
    return filename

# 각 시트별로 유사도 기반 분류 수행
def process_sheet_similarity(sheet_name, df, text_column='description', num_clusters=5):
    """각 시트별로 유사도 기반 분류를 수행하고 결과를 저장하는 함수"""
    print(f"\n\n=== {sheet_name} 시트 유사도 기반 분류 시작 ===")
    
    if len(df) == 0:
        print(f"{sheet_name} 시트에 데이터가 없습니다.")
        return None
    
    # 문서 수가 너무 적으면 클러스터 수 조정
    if len(df) < num_clusters:
        print(f"데이터 개수({len(df)})가 클러스터 수({num_clusters})보다 적습니다. 클러스터 수를 조정합니다.")
        num_clusters = max(2, len(df) // 2)
    
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
    
    # 유사도 기반 분류 수행
    try:
        # 유사도 계산
        df_clean, similarity_matrix, tfidf_matrix, vectorizer = compute_similarity_matrix(df, text_column)
        
        # 계층적 클러스터링
        clusters, Z = hierarchical_clustering(similarity_matrix, num_clusters)
        
        # 클러스터 할당
        df_clean['cluster'] = clusters
        
        # 결과 요약
        print(f"\n{sheet_name} 유사도 기반 클러스터링 결과:")
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            print(f"클러스터 {cluster}: {count}개 문서")
        
        # 시각화
        try:
            # t-SNE 시각화
            tsne_file = visualize_clusters(tfidf_matrix, clusters, sheet_name, output_dir, 'tsne')
            print(f"t-SNE 시각화 저장됨: {tsne_file}")
            
            # UMAP 시각화 (선택적)
            try:
                umap_file = visualize_clusters(tfidf_matrix, clusters, sheet_name, output_dir, 'umap')
                print(f"UMAP 시각화 저장됨: {umap_file}")
            except Exception as e:
                print(f"UMAP 시각화 중 오류 발생: {str(e)}")
            
            # 덴드로그램 시각화
            dendro_file = visualize_dendrogram(Z, sheet_name, output_dir)
            print(f"덴드로그램 저장됨: {dendro_file}")
            
            # 유사도 히트맵 시각화
            heatmap_file = visualize_similarity_heatmap(similarity_matrix, sheet_name, output_dir)
            print(f"유사도 히트맵 저장됨: {heatmap_file}")
        except Exception as e:
            print(f"시각화 중 오류 발생: {str(e)}")
        
        # CSV 파일로 저장
        output_file = os.path.join(output_dir, f"{sheet_name.replace(' ', '_').lower()}_similarity_clusters.csv")
        df_clean.to_csv(output_file, index=False, encoding='utf-8')
        print(f"분류 결과 저장됨: {output_file}")
        
        return df_clean, clusters, similarity_matrix
    
    except Exception as e:
        print(f"{sheet_name} 시트 유사도 기반 분류 중 오류 발생: {str(e)}")
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

# 각 시트별 유사도 기반 분류 수행
results = {}
for sheet_name, df in sheets_data.items():
    # 시트의 데이터 크기에 따라 클러스터 수 조정
    num_clusters = min(5, max(2, len(df) // 20)) if len(df) > 0 else 2
    result = process_sheet_similarity(sheet_name, df, num_clusters=num_clusters)
    if result is not None:
        results[sheet_name] = result

print("\n\n=== 유사도 기반 분류 완료 ===")
print(f"총 {len(results)}개 시트에 대한 유사도 기반 분류가 완료되었습니다.")

# 전체 결과 요약 보고서 작성
with open(os.path.join(output_dir, "유사도_기반_분류_결과_요약.txt"), "w", encoding="utf-8") as f:
    f.write("# 유사도 기반 분류 결과 요약\n\n")
    for sheet_name, (df_clusters, clusters, _) in results.items():
        f.write(f"\n## {sheet_name} 시트\n")
        f.write(f"- 총 문서 수: {len(df_clusters)}\n")
        f.write(f"- 클러스터 수: {len(np.unique(clusters))}\n\n")
        
        # 클러스터별 문서 수
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        f.write("### 클러스터별 문서 수\n")
        for cluster, count in cluster_counts.items():
            f.write(f"- 클러스터 {cluster}: {count}개 문서\n")
        f.write("\n")
        
        # 클러스터별 샘플 문서
        f.write("### 클러스터별 샘플 문서\n")
        for cluster in np.unique(clusters):
            cluster_docs = df_clusters[df_clusters['cluster'] == cluster]
            if len(cluster_docs) > 0:
                sample_doc = cluster_docs.iloc[0]
                if 'name' in sample_doc:
                    doc_title = sample_doc['name']
                elif 'title' in sample_doc:
                    doc_title = sample_doc['title']
                else:
                    text_col = [col for col in df_clusters.columns if 'text' in col.lower() or 'description' in col.lower()]
                    if text_col:
                        doc_title = sample_doc[text_col[0]][:50] + "..."
                    else:
                        doc_title = f"문서 #{sample_doc.name}"
                
                f.write(f"- 클러스터 {cluster} 샘플: {doc_title}\n")
        f.write("\n")

print(f"유사도 기반 분류 결과 요약 보고서가 생성되었습니다: {os.path.join(output_dir, '유사도_기반_분류_결과_요약.txt')}") 