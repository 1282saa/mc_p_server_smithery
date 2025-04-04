import pandas as pd
import numpy as np
import os
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.manifold import TSNE
import umap
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
try:
    from konlpy.tag import Mecab, Okt  # 한국어 형태소 분석기
    konlpy_available = True
except ImportError:
    konlpy_available = False
import warnings
warnings.filterwarnings('ignore')

# 결과 저장 디렉토리 생성
output_dir = "journalist_mcp_results"
os.makedirs(output_dir, exist_ok=True)

# NLTK 데이터 다운로드
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# 한국어 형태소 분석기 설정
korean_analyzer = None
if konlpy_available:
    try:
        mecab = Mecab()
        korean_analyzer = 'mecab'
    except Exception:
        try:
            okt = Okt()
            korean_analyzer = 'okt'
        except Exception:
            print("한국어 형태소 분석기(Mecab/Okt)를 로드할 수 없습니다. 기본 토큰화 방식을 사용합니다.")

# 엑셀 파일 경로
excel_file = 'smithery/smithery_mcp_server.xlsx'

print("엑셀 파일에서 데이터 로드 중...")
try:
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
    
    print("데이터 로드 완료")
    
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
    
    # 컬럼명 표준화
    if 'name' not in all_data.columns and 'search_query' in all_data.columns:
        # 두 번째 열이 name인 경우가 많음
        second_col = all_data.columns[1]
        all_data = all_data.rename(columns={second_col: 'name'})
    
    # 데이터 전처리: NaN 값 처리
    for col in ['name', '설명', 'description', 'url', 'type', 'usage_count']:
        if col in all_data.columns and all_data[col].isna().sum() > 0:
            if col in ['name', '설명', 'description', 'url']:
                all_data[col] = all_data[col].fillna('')
            elif col == 'usage_count':
                all_data[col] = all_data[col].fillna(0)
    
    # 사용 횟수 수치화 처리
    if 'usage_count' in all_data.columns:
        all_data['usage_count_numeric'] = all_data['usage_count'].apply(
            lambda x: float(str(x).replace('k', '')) * 1000 if isinstance(x, str) and 'k' in str(x)
            else float(x) if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit())
            else 0
        )
    else:
        all_data['usage_count_numeric'] = 0
    
    # 설명 텍스트 결합 (영문 + 한글)
    all_data['combined_text'] = ''
    
    for idx, row in all_data.iterrows():
        combined = []
        
        if 'name' in row and pd.notna(row['name']) and row['name'] != '':
            combined.append(str(row['name']))
            
        if 'description' in row and pd.notna(row['description']) and row['description'] != '':
            combined.append(str(row['description']))
            
        if '설명' in row and pd.notna(row['설명']) and row['설명'] != '':
            combined.append(str(row['설명']))
            
        all_data.at[idx, 'combined_text'] = ' '.join(combined)
    
    print(f"총 {len(all_data)}개의 MCP 서버 데이터 통합 및 전처리 완료")

    # 텍스트 전처리 함수
    def preprocess_text(text, language='both'):
        """
        텍스트 전처리 함수: 특수문자 제거, 소문자 변환, 한국어/영어 토큰화
        
        Args:
            text: 텍스트 문자열
            language: 'ko', 'en', 'both' 중 하나 (한국어, 영어, 또는 둘 다)
        
        Returns:
            전처리된 텍스트
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 특수문자 및 숫자 제거 (영어/한글/공백만 유지)
        text = re.sub(r'[^\w\s가-힣]', ' ', text.lower())
        text = re.sub(r'\d+', ' ', text)
        
        # 여러 공백을 하나로 치환
        text = re.sub(r'\s+', ' ', text).strip()
        
        if language == 'ko' and korean_analyzer:
            # 한국어 형태소 분석
            if korean_analyzer == 'mecab':
                tokens = mecab.morphs(text)
            else:  # okt
                tokens = okt.morphs(text)
            return ' '.join(tokens)
        
        elif language == 'en':
            # 영어 토큰화
            tokens = nltk.word_tokenize(text)
            return ' '.join(tokens)
        
        else:  # both - 기본 처리만 수행
            return text
    
    # 전체 텍스트에 전처리 적용
    all_data['processed_text'] = all_data['combined_text'].apply(lambda x: preprocess_text(x))
    
    # 기자/신문사 업무 관련 핵심 키워드 정의 (대폭 확장)
    journalism_keywords = {
        # 1. 뉴스 보도 및 작성 관련
        "뉴스 작성": ["news writing", "article creation", "content creation", "story creation", "reporting", 
                    "기사 작성", "뉴스 작성", "기사 보도", "콘텐츠 제작", "기사 생성", "스토리 작성", "보도",
                    "editorial", "headline", "journalistic writing", "newsroom", "사설", "헤드라인", "제목",
                    "편집", "보도자료", "기사문", "원고", "뉴스룸", "데스킹", "꼭지", "타이틀", "리드문"],
        
        # 2. 취재 및 인터뷰 관련
        "취재 및 인터뷰": ["interview", "investigation", "research", "source", "background check",
                    "인터뷰", "취재", "조사", "탐사", "현장 보도", "배경 조사", "정보원", "취재원",
                    "on-site reporting", "field reporting", "press conference", "현장 취재", "기자회견",
                    "녹취", "질문", "증언", "목격자", "발언", "현장", "증거 수집", "녹음", "녹화"],
        
        # 3. 팩트체크 및 검증
        "팩트체크": ["fact check", "verification", "accuracy", "validate", "credibility", "authenticate",
                "팩트체크", "검증", "사실 확인", "정확성", "신뢰성", "진위 판별", "검증 도구",
                "source verification", "claim verification", "debunking", "출처 확인", "주장 검증",
                "오보", "진실 여부", "거짓 정보", "허위 정보", "날조", "왜곡", "참고문헌", "인증"],
        
        # 4. 데이터 저널리즘
        "데이터 저널리즘": ["data journalism", "data analysis", "statistics", "visualization", "data mining",
                       "데이터 저널리즘", "데이터 분석", "통계", "시각화", "데이터 마이닝", "차트", "그래프",
                       "infographic", "chart", "graph", "dataset", "인포그래픽", "데이터셋", "수치 분석",
                       "빅데이터", "데이터 시각화", "데이터 수집", "패턴 분석", "계량 분석", "측정"],
        
        # 5. 실시간 뉴스 및 속보
        "실시간 뉴스": ["breaking news", "live update", "real-time", "instant", "alert", "latest",
                    "속보", "실시간", "뉴스 알림", "긴급 뉴스", "최신 소식", "업데이트",
                    "fast journalism", "rapid reporting", "timely news", "빠른 보도", "신속 보도",
                    "실시간 전달", "뉴스 피드", "최신 정보", "경보", "알림", "긴급", "최신"],
        
        # 6. 트렌드 및 이슈 분석
        "트렌드 분석": ["trend analysis", "issue tracking", "social monitoring", "buzz tracking", "viral content",
                    "트렌드 분석", "이슈 추적", "소셜 모니터링", "화제성 추적", "바이럴 콘텐츠",
                    "popular topics", "public opinion", "hot issues", "인기 주제", "여론", "화제",
                    "시사", "핫이슈", "키워드 분석", "관심사", "트렌딩", "유행", "흐름", "동향", "인기"],
        
        # 7. 소셜미디어 및 디지털 콘텐츠
        "소셜미디어": ["social media", "digital content", "online journalism", "social networks", "viral",
                   "소셜미디어", "디지털 콘텐츠", "온라인 저널리즘", "소셜 네트워크", "바이럴",
                   "twitter", "facebook", "instagram", "youtube", "tiktok", "트위터", "페이스북",
                   "유튜브", "인스타그램", "틱톡", "SNS", "댓글", "공유", "좋아요", "전파"],
        
        # 8. 멀티미디어 콘텐츠
        "멀티미디어": ["multimedia", "photo", "video", "audio", "podcast", "visualization",
                   "멀티미디어", "사진", "동영상", "오디오", "팟캐스트", "시각화",
                   "image processing", "video editing", "sound editing", "이미지 처리", "영상 편집",
                   "음향 편집", "그래픽", "인포그래픽", "사진 보정", "촬영", "편집", "음성"],
        
        # 9. 국제 뉴스 및 번역
        "국제 뉴스": ["international news", "global", "foreign correspondence", "translation", "multilingual",
                   "국제 뉴스", "글로벌", "해외 통신", "번역", "다국어", "외신", "국제 정세",
                   "cross-border", "world affairs", "foreign press", "국경간", "세계 정세", "해외 언론",
                   "국제 보도", "해외 특파원", "현지 소식", "각국", "세계", "국가간", "국제"],
        
        # 10. 아카이브 및 검색
        "아카이브": ["archive", "search", "database", "repository", "historical data", "library",
                 "아카이브", "검색", "데이터베이스", "리포지토리", "역사적 데이터", "라이브러리",
                 "information retrieval", "cataloging", "indexing", "정보 검색", "카탈로그", "색인",
                 "저장소", "기록 보관", "보존", "수집", "자료", "과거 기사", "스크랩"],
        
        # 11. AI 저널리즘 및 자동화
        "AI 저널리즘": ["ai journalism", "automated journalism", "robot reporting", "nlg", "ai writing",
                    "AI 저널리즘", "자동화 저널리즘", "로봇 기자", "자연어 생성", "AI 글쓰기",
                    "machine learning journalism", "automated content", "기계학습 저널리즘", "자동화 콘텐츠",
                    "AI 분석", "자동 요약", "자동 번역", "자동 생성", "알고리즘 기사", "컴퓨터 기자"],
        
        # 12. 뉴스 큐레이션 및 개인화
        "뉴스 큐레이션": ["news curation", "personalization", "recommendation", "tailored news", "custom feed",
                     "뉴스 큐레이션", "개인화", "추천", "맞춤형 뉴스", "커스텀 피드",
                     "content filtering", "user preference", "targeted content", "콘텐츠 필터링", "사용자 선호도",
                     "맞춤 콘텐츠", "선별", "분류", "편집", "맞춤", "추천 시스템"],
        
        # 13. 웹 스크래핑 및 데이터 수집
        "웹 스크래핑": ["web scraping", "data extraction", "crawling", "web automation", "data gathering",
                    "웹 스크래핑", "데이터 추출", "크롤링", "웹 자동화", "데이터 수집",
                    "information extraction", "web mining", "정보 추출", "웹 마이닝", "크롤러",
                    "데이터 수확", "자동 수집", "정보 수집", "자동 추출", "웹 데이터"],
        
        # 14. 뉴스 배포 및 타겟팅
        "뉴스 배포": ["news distribution", "publishing", "media outlet", "circulation", "broadcasting",
                  "뉴스 배포", "출판", "미디어 아웃렛", "유통", "방송",
                  "syndication", "news feed", "content delivery", "신디케이션", "뉴스 피드",
                  "콘텐츠 전달", "배포", "전송", "공유", "홍보", "퍼블리싱"]
    }

    # 모든 키워드를 하나의 리스트로 통합 (중복 제거)
    all_journalism_keywords = set()
    for category_keywords in journalism_keywords.values():
        all_journalism_keywords.update([kw.lower() for kw in category_keywords])
    
    all_journalism_keywords = list(all_journalism_keywords)
    print(f"총 {len(all_journalism_keywords)}개의 기자 업무 관련 키워드 정의 완료")

    # 키워드 기반 검색 함수
    def find_keyword_matches(df, text_col, keywords, min_score=1):
        """
        텍스트에서 키워드 매칭 점수를 계산하는 함수
        
        Args:
            df: 데이터프레임
            text_col: 텍스트가 있는 열 이름
            keywords: 검색할 키워드 목록
            min_score: 최소 점수 (이 점수 이상인 항목만 반환)
            
        Returns:
            점수가 포함된 데이터프레임
        """
        # 결측치 처리
        valid_df = df.dropna(subset=[text_col]).copy()
        
        # 각 MCP 서버의 설명에서 키워드 점수 계산
        scores = []
        matched_keywords = []
        
        for idx, row in valid_df.iterrows():
            text = str(row[text_col]).lower()
            
            # 각 키워드가 텍스트에 포함되어 있는지 확인하고 일치하는 키워드 저장
            found_keywords = []
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in text:
                    found_keywords.append(keyword)
            
            # 일치하는 키워드 수 저장
            scores.append(len(found_keywords))
            matched_keywords.append(found_keywords)
        
        # 결과 데이터프레임에 점수와 일치 키워드 추가
        valid_df['keyword_score'] = scores
        valid_df['matched_keywords'] = matched_keywords
        
        # 최소 점수 이상인 항목만 반환
        result_df = valid_df[valid_df['keyword_score'] >= min_score].sort_values('keyword_score', ascending=False)
        
        return result_df
    
    # 각 카테고리별 관련 MCP 서버 찾기
    def find_category_related_servers(df, category, keywords, text_col='combined_text', top_n=10):
        """
        특정 카테고리와 관련 있는 MCP 서버를 찾는 함수
        
        Args:
            df: 데이터프레임
            category: 카테고리 이름
            keywords: 해당 카테고리 키워드 목록
            text_col: 텍스트가 있는 열 이름
            top_n: 상위 결과 수
            
        Returns:
            관련 서버 데이터프레임
        """
        # 카테고리 관련 키워드로 매칭 찾기
        related_servers = find_keyword_matches(df, text_col, keywords)
        
        # 결과 정렬 및 상위 N개 반환
        top_servers = related_servers.sort_values(['keyword_score', 'usage_count_numeric'], 
                                                ascending=[False, False]).head(top_n)
        
        return top_servers
    
    # 기자 업무 관련 복합 점수 계산 함수
    def calculate_journalism_scores(df, text_col='combined_text'):
        """
        모든 기자 관련 카테고리에 대한 점수를 계산하는 함수
        
        Args:
            df: 데이터프레임
            text_col: 텍스트가 있는 열 이름
            
        Returns:
            카테고리별 점수와 총점이 추가된 데이터프레임
        """
        result_df = df.copy()
        
        # 각 카테고리별 점수 계산
        for category, keywords in journalism_keywords.items():
            category_col = f"{category}_score"
            
            # 각 항목의 해당 카테고리 점수 계산
            scores = []
            category_keywords = []
            
            for idx, row in result_df.iterrows():
                if pd.notna(row[text_col]):
                    text = str(row[text_col]).lower()
                    found_keywords = []
                    
                    for keyword in keywords:
                        if keyword.lower() in text:
                            found_keywords.append(keyword)
                    
                    scores.append(len(found_keywords))
                    category_keywords.append(found_keywords)
                else:
                    scores.append(0)
                    category_keywords.append([])
            
            result_df[category_col] = scores
            result_df[f"{category}_keywords"] = category_keywords
        
        # 총점 계산 (모든 카테고리 점수의 합)
        score_columns = [f"{cat}_score" for cat in journalism_keywords.keys()]
        result_df['total_journalism_score'] = result_df[score_columns].sum(axis=1)
        
        # 점수 정규화 (Min-Max 스케일링, 0-10 범위)
        max_score = result_df['total_journalism_score'].max()
        min_score = result_df['total_journalism_score'].min()
        
        if max_score > min_score:
            result_df['journalism_score_normalized'] = 10 * (result_df['total_journalism_score'] - min_score) / (max_score - min_score)
        else:
            result_df['journalism_score_normalized'] = 5  # 모든 점수가 같을 경우 중간값 부여
            
        # 사용량을 고려한 종합 점수
        # 사용량 정규화 (0-10 범위)
        max_usage = result_df['usage_count_numeric'].max()
        min_usage = result_df['usage_count_numeric'].min()
        
        if max_usage > min_usage:
            result_df['usage_normalized'] = 10 * (result_df['usage_count_numeric'] - min_usage) / (max_usage - min_usage)
        else:
            result_df['usage_normalized'] = 5
        
        # 종합 점수 계산 (기자 관련성 70%, 사용량 30%)
        result_df['composite_score'] = 0.7 * result_df['journalism_score_normalized'] + 0.3 * result_df['usage_normalized']
        
        # 주요 기자 업무 카테고리 태그 추가
        result_df['primary_journalism_category'] = ''
        for idx, row in result_df.iterrows():
            # 가장 높은 점수를 가진 카테고리 찾기
            category_scores = {cat: row[f"{cat}_score"] for cat in journalism_keywords.keys()}
            if any(category_scores.values()):  # 점수가 하나라도 있으면
                top_category = max(category_scores.items(), key=lambda x: x[1])[0]
                result_df.at[idx, 'primary_journalism_category'] = top_category
        
        return result_df

except Exception as e:
    print(f"데이터 로드 중 오류 발생: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1) 