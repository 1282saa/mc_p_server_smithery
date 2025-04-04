# 엑셀 시트 데이터 분류 도구

이 프로젝트는 엑셀 파일의 여러 시트에 담긴 데이터를 자동으로 분류하고 시각화하는 도구입니다. 텍스트 기반 분류, 토픽 모델링, 유사도 기반 클러스터링 등 다양한 방법으로 데이터를 분석합니다.

## 기능

- **K-means 클러스터링 기반 분류**: TF-IDF 벡터화와 K-means 클러스터링을 사용하여 텍스트 데이터를 분류합니다.
- **토픽 모델링(LDA) 기반 분류**: 잠재 디리클레 할당(LDA) 알고리즘을 사용하여 텍스트에서 주제를 추출합니다.
- **유사도 기반 계층적 클러스터링**: 문서 간 유사도 행렬을 바탕으로 계층적 클러스터링을 수행합니다.
- **다양한 시각화**: 워드클라우드, 토픽 분포, 클러스터 시각화(t-SNE, UMAP), 덴드로그램, 유사도 히트맵 등

## 시작하기

### 요구사항

- Python 3.6 이상
- pandas, numpy, scikit-learn, matplotlib, seaborn, wordcloud, umap-learn, openpyxl 패키지

### 설치

```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud umap-learn openpyxl
```

### 실행 방법

1. 분석할 엑셀 파일(`smithery_mcp_server.xlsx`)을 프로젝트 디렉토리에 위치시킵니다.
2. 다음 명령어로 통합 분류 도구를 실행합니다:

```bash
python run_classification.py
```

또는 개별 분류 스크립트를 직접 실행할 수 있습니다:

```bash
# K-means 클러스터링 기반 분류
python classify_data.py

# 토픽 모델링(LDA) 기반 분류
python topic_modeling.py

# 유사도 기반 계층적 클러스터링
python similarity_clustering.py
```

## 결과 확인

분류 결과는 다음 디렉토리에 저장됩니다:

- `classified_results/`: K-means 클러스터링 결과
- `topic_modeling_results/`: 토픽 모델링 결과 및 시각화
- `similarity_clustering_results/`: 유사도 기반 클러스터링 결과 및 시각화
- `classification_results/`: 통합 결과 디렉토리

각 디렉토리에는 분류된 CSV 파일과 시각화 결과, 그리고 요약 보고서가 포함됩니다.

## 파일 설명

- `run_classification.py`: 통합 실행 스크립트
- `classify_data.py`: K-means 클러스터링 기반 분류 스크립트
- `topic_modeling.py`: 토픽 모델링(LDA) 기반 분류 스크립트
- `similarity_clustering.py`: 유사도 기반 계층적 클러스터링 스크립트
- `read_excel.py`: 엑셀 파일 읽기 및 정보 출력 스크립트
- `convert_excel.py`: 엑셀 파일을 CSV로 변환하는 스크립트

## 분류 방법 상세 설명

### 1. K-means 클러스터링 기반 분류

텍스트 데이터를 TF-IDF(Term Frequency-Inverse Document Frequency) 벡터로 변환한 후, K-means 알고리즘을 사용하여 유사한 문서들을 그룹화합니다. 각 클러스터의 주요 키워드를 추출하여 클러스터의 특성을 파악할 수 있습니다.

### 2. 토픽 모델링(LDA) 기반 분류

LDA(Latent Dirichlet Allocation) 알고리즘을 사용하여 텍스트에 숨겨진 주제(토픽)를 찾아냅니다. 각 문서가 여러 토픽의 혼합으로 구성되어 있다고 가정하고, 토픽별 단어 분포와 문서별 토픽 분포를 추출합니다. 워드클라우드를 통해 각 토픽의 주요 단어를 시각적으로 확인할 수 있습니다.

### 3. 유사도 기반 계층적 클러스터링

문서 간 코사인 유사도를 계산하여 유사도 행렬을 생성한 후, 계층적 클러스터링 알고리즘을 적용합니다. 덴드로그램을 통해 문서 간 계층 구조를 시각화하고, t-SNE나 UMAP을 통해 고차원 데이터를 2차원으로 축소하여 클러스터를 시각화합니다.

## 커스터마이징

각 스크립트의 파라미터를 수정하여 분류 방법을 조정할 수 있습니다:

- 클러스터/토픽 수 조정
- 사용할 텍스트 컬럼 변경
- 시각화 옵션 수정
- 전처리 방법 조정

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.
