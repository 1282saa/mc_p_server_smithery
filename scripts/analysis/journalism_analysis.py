"""
기자 업무 분석을 위한 MCP 서버 분석 통합 스크립트

이 스크립트는 다음 기능들을 통합합니다:
1. 단순 키워드 기반 기자 관련 MCP 서버 검색 (simple_news_mcp_analyzer.py)
2. 뉴스 관련 MCP 서버 분석 (news_focused_mcp_analyzer.py)
3. 기자 업무별 고급 분석 및 추천 (journalist_mcp_analyzer.py)

사용법:
    python journalism_analysis.py --mode simple|basic|advanced
"""

import pandas as pd
import numpy as np
import os
import re
import argparse
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 필요 시 추가 라이브러리 임포트
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk_available = True
except ImportError:
    nltk_available = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans, AgglomerativeClustering
    scikit_available = True
except ImportError:
    scikit_available = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    viz_available = True
except ImportError:
    viz_available = False

try:
    from konlpy.tag import Mecab, Okt
    konlpy_available = True
except ImportError:
    konlpy_available = False


class JournalismMCPAnalyzer:
    """기자 업무를 위한 MCP 서버 분석 클래스"""
    
    def __init__(self, excel_path='../data/smithery_mcp_server.xlsx', output_dir='../results/journalism_results'):
        """
        초기화 함수
        
        Args:
            excel_path: MCP 서버 데이터가 있는 엑셀 파일 경로
            output_dir: 결과 저장 디렉토리
        """
        self.excel_path = excel_path
        self.output_dir = output_dir
        self.all_data = None
        self.korean_analyzer = None
        
        # 결과 저장 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # NLTK 준비 (필요시)
        if nltk_available:
            try:
                nltk.data.find('corpora/stopwords')
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('stopwords')
                nltk.download('punkt')
        
        # 한국어 형태소 분석기 설정
        if konlpy_available:
            try:
                self.mecab = Mecab()
                self.korean_analyzer = 'mecab'
            except Exception:
                try:
                    self.okt = Okt()
                    self.korean_analyzer = 'okt'
                except Exception:
                    print("한국어 형태소 분석기(Mecab/Okt)를 로드할 수 없습니다. 기본 토큰화 방식을 사용합니다.")
        
        # 기자/신문사 업무 관련 핵심 키워드 정의
        self.journalism_keywords = {
            # 1. 뉴스 보도 및 작성 관련
            "뉴스 작성": ["news writing", "article creation", "content creation", "reporting", 
                       "기사 작성", "뉴스 작성", "기사 보도", "콘텐츠 제작", "headline", "제목"],
            
            # 2. 취재 및 인터뷰 관련
            "취재 및 인터뷰": ["interview", "investigation", "research", "source", 
                        "인터뷰", "취재", "조사", "탐사", "정보원", "취재원"],
            
            # 3. 팩트체크 및 검증
            "팩트체크": ["fact check", "verification", "accuracy", "credibility", 
                    "팩트체크", "검증", "사실 확인", "정확성", "신뢰성", "출처 확인"],
            
            # 4. 데이터 저널리즘
            "데이터 저널리즘": ["data journalism", "data analysis", "statistics", "visualization", 
                          "데이터 저널리즘", "데이터 분석", "통계", "시각화", "차트", "그래프"],
            
            # 5. 실시간 뉴스 및 속보
            "실시간 뉴스": ["breaking news", "real-time", "alert", "latest",
                       "속보", "실시간", "뉴스 알림", "긴급 뉴스", "최신 소식"],
            
            # 6. 트렌드 및 이슈 분석
            "트렌드 분석": ["trend analysis", "issue tracking", "social monitoring", 
                       "트렌드 분석", "이슈 추적", "소셜 모니터링", "인기 주제", "여론"],
            
            # 7. 소셜미디어 및 디지털 콘텐츠
            "소셜미디어": ["social media", "digital content", "online journalism", 
                      "소셜미디어", "디지털 콘텐츠", "온라인 저널리즘", "SNS"],
            
            # 8. 멀티미디어 콘텐츠
            "멀티미디어": ["multimedia", "photo", "video", "audio", "podcast",
                      "멀티미디어", "사진", "동영상", "오디오", "팟캐스트"]
        }
        
        # 모든 키워드 통합
        self.all_keywords = []
        for category_keywords in self.journalism_keywords.values():
            self.all_keywords.extend(category_keywords)
    
    def load_data(self):
        """엑셀 파일에서 데이터 로드 및 전처리"""
        print(f"엑셀 파일({self.excel_path})에서 데이터 로드 중...")
        
        try:
            # 모든 시트 데이터 로드
            excel_file = pd.ExcelFile(self.excel_path)
            
            # 시트 이름 확인
            print(f"엑셀 파일 시트: {excel_file.sheet_names}")
            
            # 모든 데이터프레임을 하나로 합치기
            dfs = []
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, skiprows=3,
                                     usecols=lambda x: not x.startswith('Unnamed: 0'))
                    
                    if len(df) > 0:
                        df['sheet_name'] = sheet_name
                        dfs.append(df)
                except Exception as e:
                    print(f"시트 {sheet_name} 로딩 중 오류: {str(e)}")
            
            # 전체 데이터프레임 만들기
            self.all_data = pd.concat(dfs, ignore_index=True)
            
            # 컬럼명 표준화
            if 'name' not in self.all_data.columns and self.all_data.shape[1] > 1:
                # 두 번째 열이 name인 경우가 많음
                second_col = self.all_data.columns[1]
                self.all_data = self.all_data.rename(columns={second_col: 'name'})
            
            # 데이터 전처리: NaN 값 처리
            for col in ['name', '설명', 'description', 'url', 'type', 'usage_count']:
                if col in self.all_data.columns and self.all_data[col].isna().sum() > 0:
                    if col in ['name', '설명', 'description', 'url']:
                        self.all_data[col] = self.all_data[col].fillna('')
                    elif col == 'usage_count':
                        self.all_data[col] = self.all_data[col].fillna(0)
            
            # 사용 횟수 수치화 처리
            if 'usage_count' in self.all_data.columns:
                self.all_data['usage_count_numeric'] = self.all_data['usage_count'].apply(
                    lambda x: float(str(x).replace('k', '')) * 1000 if isinstance(x, str) and 'k' in str(x)
                    else float(x) if isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '', 1).isdigit())
                    else 0
                )
            else:
                self.all_data['usage_count_numeric'] = 0
            
            # 설명 텍스트 결합 (영문 + 한글)
            self.all_data['combined_text'] = ''
            for idx, row in self.all_data.iterrows():
                combined = []
                
                if 'name' in row and pd.notna(row['name']) and row['name'] != '':
                    combined.append(str(row['name']))
                    
                if 'description' in row and pd.notna(row['description']) and row['description'] != '':
                    combined.append(str(row['description']))
                    
                if '설명' in row and pd.notna(row['설명']) and row['설명'] != '':
                    combined.append(str(row['설명']))
                    
                self.all_data.at[idx, 'combined_text'] = ' '.join(combined)
            
            print(f"총 {len(self.all_data)}개의 MCP 서버 데이터 로드 및 전처리 완료")
            return True
            
        except Exception as e:
            print(f"데이터 로드 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def find_journalism_mcps_simple(self, top_n=50):
        """
        간단한 키워드 매칭 방식으로 기자 관련 MCP 서버 찾기 (단순 모드)
        
        Args:
            top_n: 상위 결과 수
            
        Returns:
            상위 MCP 서버 데이터프레임
        """
        print("\n1. 기자/신문사 관련 MCP 서버 탐색 중 (단순 모드)...")
        
        # 결측치 처리
        df = self.all_data.dropna(subset=['combined_text']).copy()
        
        # 각 MCP 서버의 설명에서 키워드 점수 계산
        scores = []
        matched_keywords = []
        
        for idx, row in df.iterrows():
            text = str(row['combined_text']).lower()
            found_keywords = []
            
            for keyword in self.all_keywords:
                if keyword.lower() in text:
                    found_keywords.append(keyword)
            
            scores.append(len(found_keywords))
            matched_keywords.append(', '.join(found_keywords))
        
        # 결과 데이터프레임에 점수와 일치 키워드 추가
        df['relevance_score'] = scores
        df['matched_keywords'] = matched_keywords
        
        # 점수 기준 내림차순 정렬
        result_df = df.sort_values('relevance_score', ascending=False)
        
        # 상위 N개 결과 추출
        top_results = result_df.head(top_n)
        
        # 결과 저장
        result_file = os.path.join(self.output_dir, "journalism_mcps_simple_top50.csv")
        top_results.to_csv(result_file, index=False, encoding='utf-8-sig')
        print(f"기자 관련 상위 {top_n}개 MCP 서버 목록 저장 완료: {result_file}")
        
        # 시트별 분포 계산
        sheet_counts = top_results['sheet_name'].value_counts()
        print("\n상위 MCP 서버의 시트별 분포:")
        for sheet, count in sheet_counts.items():
            print(f"- {sheet}: {count}개")
        
        return top_results
    
    def find_journalism_mcps_by_category(self, top_n=10):
        """
        카테고리별로 기자 관련 MCP 서버 찾기 (기본 모드)
        
        Args:
            top_n: 각 카테고리별 상위 결과 수
            
        Returns:
            카테고리별 결과를 담은 사전
        """
        print("\n2. 기자 업무 카테고리별 MCP 서버 찾기 (기본 모드)...")
        
        category_results = {}
        
        for category, keywords in self.journalism_keywords.items():
            print(f"\n[{category}] 관련 MCP 서버 찾는 중...")
            
            # 결측치 처리
            df = self.all_data.dropna(subset=['combined_text']).copy()
            
            # 키워드 매칭 점수 계산
            scores = []
            matched_keywords = []
            
            for idx, row in df.iterrows():
                text = str(row['combined_text']).lower()
                found_keywords = []
                
                for keyword in keywords:
                    if keyword.lower() in text:
                        found_keywords.append(keyword)
                
                scores.append(len(found_keywords))
                matched_keywords.append(', '.join(found_keywords))
            
            # 결과 데이터프레임에 점수와 일치 키워드 추가
            df[f'{category}_score'] = scores
            df[f'{category}_keywords'] = matched_keywords
            
            # 점수 기준 내림차순 정렬
            result_df = df.sort_values([f'{category}_score', 'usage_count_numeric'],
                                      ascending=[False, False])
            
            # 상위 N개 결과 추출
            top_results = result_df[result_df[f'{category}_score'] > 0].head(top_n)
            
            # 결과 저장
            category_file = os.path.join(self.output_dir, 
                                        f"{category.replace(' ', '_')}_추천_MCP서버.csv")
            if len(top_results) > 0:
                top_results.to_csv(category_file, index=False, encoding='utf-8-sig')
                print(f"  '{category}' 관련 상위 {len(top_results)}개 MCP 서버 목록 저장 완료")
                
                # 상위 3개 서버 출력
                for i, (_, row) in enumerate(top_results.head(3).iterrows()):
                    print(f"    {i+1}. {row.get('name', '')}: {row.get('설명', '')[:100]}...")
                
                category_results[category] = top_results
            else:
                print(f"  '{category}' 관련 MCP 서버를 찾을 수 없습니다.")
        
        return category_results
    
    def find_journalism_mcps_advanced(self):
        """
        고급 분석을 통한 기자 관련 MCP 서버 찾기 (고급 모드)
        """
        print("\n3. 고급 분석을 통한 기자 관련 MCP 서버 찾기 (고급 모드)...")
        
        if not scikit_available:
            print("고급 분석에 필요한 scikit-learn 라이브러리가 설치되어 있지 않습니다.")
            print("pip install scikit-learn 명령어로 설치해주세요.")
            return None
        
        # TF-IDF 벡터화
        print("텍스트 임베딩 생성 중...")
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        
        # 설명 열이 비어 있지 않은 행만 사용
        valid_df = self.all_data.dropna(subset=['combined_text']).copy()
        
        # TF-IDF 행렬 생성
        tfidf_matrix = tfidf.fit_transform(valid_df['combined_text'].astype(str))
        
        # 기자 업무 관련 쿼리 생성
        journalism_query = """
        뉴스 기사 작성, 취재, 인터뷰, 팩트체크, 데이터 분석, 뉴스 속보, 트렌드 분석, 
        소셜미디어 모니터링, 멀티미디어 콘텐츠 제작, 국제 뉴스 번역, 아카이브 검색, 
        AI 기사 작성 지원, 뉴스 큐레이션, 웹 스크래핑, 뉴스 배포 등의 기자 업무를 
        효율적으로 수행할 수 있도록 도와주는 도구
        """
        
        # 쿼리 벡터 생성
        query_vec = tfidf.transform([journalism_query])
        
        # 코사인 유사도 계산
        cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # 유사도 점수 추가
        valid_df['similarity_score'] = cosine_similarities
        
        # 종합 점수 계산 (유사도 70%, 사용량 30%)
        # 사용량 정규화
        max_usage = valid_df['usage_count_numeric'].max()
        min_usage = valid_df['usage_count_numeric'].min()
        
        if max_usage > min_usage:
            valid_df['usage_normalized'] = (valid_df['usage_count_numeric'] - min_usage) / (max_usage - min_usage)
        else:
            valid_df['usage_normalized'] = 0.5
        
        # 유사도 정규화
        max_sim = valid_df['similarity_score'].max()
        min_sim = valid_df['similarity_score'].min()
        
        if max_sim > min_sim:
            valid_df['similarity_normalized'] = (valid_df['similarity_score'] - min_sim) / (max_sim - min_sim)
        else:
            valid_df['similarity_normalized'] = 0.5
        
        # 종합 점수 계산
        valid_df['composite_score'] = 0.7 * valid_df['similarity_normalized'] + 0.3 * valid_df['usage_normalized']
        
        # 점수 기준 내림차순 정렬
        result_df = valid_df.sort_values('composite_score', ascending=False)
        
        # 상위 30개 결과 추출
        top_results = result_df.head(30)
        
        # 결과 저장
        result_file = os.path.join(self.output_dir, "journalism_mcps_advanced_top30.csv")
        top_results.to_csv(result_file, index=False, encoding='utf-8-sig')
        print(f"고급 분석 결과 상위 30개 MCP 서버 목록 저장 완료: {result_file}")
        
        # 상위 5개 서버 출력
        print("\n고급 분석 결과 상위 5개 MCP 서버:")
        for i, (_, row) in enumerate(top_results.head(5).iterrows()):
            print(f"{i+1}. {row.get('name', '')}: {row.get('설명', '')[:100]}...")
            print(f"   URL: {row.get('url', '')}")
            print(f"   카테고리: {row.get('sheet_name', '')}")
            print(f"   유사도 점수: {row.get('similarity_score', 0):.4f}")
            print(f"   종합 점수: {row.get('composite_score', 0):.4f}")
            print("")
        
        return top_results
    
    def run_analysis(self, mode='basic'):
        """
        지정된 모드에 따라 분석 실행
        
        Args:
            mode: 'simple', 'basic', 'advanced' 중 하나
        """
        if not self.load_data():
            print("데이터 로드에 실패했습니다. 분석을 중단합니다.")
            return
        
        if mode == 'simple':
            self.find_journalism_mcps_simple()
        elif mode == 'basic':
            self.find_journalism_mcps_simple()
            self.find_journalism_mcps_by_category()
        elif mode == 'advanced':
            self.find_journalism_mcps_simple()
            self.find_journalism_mcps_by_category()
            self.find_journalism_mcps_advanced()
        else:
            print(f"잘못된 모드입니다: {mode}")
            print("사용 가능한 모드: simple, basic, advanced")
            return
        
        print("\n분석이 완료되었습니다. 결과를 확인하세요.")
        print(f"결과 파일 위치: {self.output_dir}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='기자 업무를 위한 MCP 서버 분석 도구')
    parser.add_argument('--mode', type=str, default='basic',
                      choices=['simple', 'basic', 'advanced'],
                      help='분석 모드 (simple/basic/advanced)')
    parser.add_argument('--excel', type=str, default='../data/smithery_mcp_server.xlsx',
                      help='엑셀 파일 경로')
    parser.add_argument('--output', type=str, default='../results/journalism_results',
                      help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 분석기 초기화 및 실행
    analyzer = JournalismMCPAnalyzer(excel_path=args.excel, output_dir=args.output)
    analyzer.run_analysis(mode=args.mode)


if __name__ == "__main__":
    main() 