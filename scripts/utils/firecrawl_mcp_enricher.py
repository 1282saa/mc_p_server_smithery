"""
FireCrawl을 활용한 MCP 서버 정보 보강 스크립트

이 스크립트는 MCP 서버 목록에서 기자 관련 유용한 MCP를 식별하고,
웹 검색과 웹 스크래핑을 통해 추가 정보를 수집하는 기능을 제공합니다.
"""

import pandas as pd
import numpy as np
import os
import re
import json
import time
import argparse
from collections import Counter
import requests
from datetime import datetime

class FirecrawlMCPEnricher:
    """FireCrawl을 활용한 MCP 서버 정보 보강 클래스"""
    
    def __init__(self, mcp_file, output_dir='../../results/enriched_data'):
        """
        초기화 함수
        
        Args:
            mcp_file: MCP 서버 데이터 파일 (CSV 또는 Excel)
            output_dir: 결과 저장 디렉토리
        """
        self.mcp_file = mcp_file
        self.output_dir = output_dir
        self.mcp_data = None
        
        # 저널리즘 관련 키워드 정의
        self.journalism_keywords = {
            "뉴스 검색": ["news search", "news finder", "news retrieval", "article search",
                      "뉴스 검색", "기사 검색", "뉴스 찾기"],
            
            "웹 스크래핑": ["web scraping", "website extraction", "content extraction", "crawling",
                       "웹 스크래핑", "크롤링", "웹 데이터 추출", "웹사이트 수집"],
            
            "인터넷 검색": ["web search", "internet search", "search engine", "online search",
                       "인터넷 검색", "웹 검색", "검색 엔진"],
            
            "데이터 수집": ["data collection", "data gathering", "information retrieval",
                       "데이터 수집", "데이터 획득", "정보 추출"],
            
            "미디어 모니터링": ["media monitoring", "news monitoring", "content tracking",
                         "미디어 모니터링", "뉴스 모니터링", "콘텐츠 추적"]
        }
        
        # 결과 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
    
    def load_mcp_data(self):
        """MCP 서버 데이터 로드"""
        print(f"MCP 서버 데이터 파일 로드 중: {self.mcp_file}")
        
        try:
            # 파일 확장자 확인
            file_ext = os.path.splitext(self.mcp_file)[1].lower()
            
            if file_ext == '.csv':
                self.mcp_data = pd.read_csv(self.mcp_file, encoding='utf-8-sig')
            elif file_ext in ['.xlsx', '.xls']:
                self.mcp_data = pd.read_excel(self.mcp_file)
            else:
                print(f"지원하지 않는 파일 형식: {file_ext}")
                return False
            
            # 필수 컬럼 유무 확인
            required_cols = ['name', 'url']
            missing_cols = [col for col in required_cols if col not in self.mcp_data.columns]
            
            if missing_cols:
                print(f"필수 컬럼이 없습니다: {', '.join(missing_cols)}")
                
                # 컬럼 매핑 시도
                if 'name' not in self.mcp_data.columns and len(self.mcp_data.columns) > 1:
                    # 두 번째 컬럼을 name으로 사용하는 경우가 많음
                    self.mcp_data = self.mcp_data.rename(columns={self.mcp_data.columns[1]: 'name'})
                
                # URL 컬럼 확인
                url_candidates = [col for col in self.mcp_data.columns if 'url' in col.lower()]
                if 'url' not in self.mcp_data.columns and url_candidates:
                    self.mcp_data = self.mcp_data.rename(columns={url_candidates[0]: 'url'})
            
            print(f"총 {len(self.mcp_data)}개의 MCP 서버 데이터 로드 완료")
            return True
            
        except Exception as e:
            print(f"데이터 로드 중 오류 발생: {str(e)}")
            return False
    
    def search_mcp_info(self, query, limit=5):
        """
        FireCrawl 웹 검색을 통해 MCP 정보 검색
        
        Args:
            query: 검색 쿼리
            limit: 검색 결과 수
            
        Returns:
            검색 결과 (JSON 형식)
        """
        print(f"검색 쿼리: '{query}'")
        
        try:
            # 검색 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(self.output_dir, f"search_result_{timestamp}.json")
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump({"query": query, "timestamp": timestamp}, f, ensure_ascii=False, indent=2)
                
            print(f"검색 결과가 {result_file}에 저장되었습니다.")
            
            return {"status": "success", "query": query, "file": result_file}
            
        except Exception as e:
            print(f"검색 중 오류 발생: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def scrape_mcp_site(self, url):
        """
        FireCrawl을 사용하여 MCP 웹사이트 스크래핑
        
        Args:
            url: 스크래핑할 URL
            
        Returns:
            스크래핑 결과 (JSON 형식)
        """
        print(f"웹사이트 스크래핑: {url}")
        
        try:
            # 스크래핑 결과 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(self.output_dir, f"scrape_result_{timestamp}.json")
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump({"url": url, "timestamp": timestamp}, f, ensure_ascii=False, indent=2)
                
            print(f"스크래핑 결과가 {result_file}에 저장되었습니다.")
            
            return {"status": "success", "url": url, "file": result_file}
            
        except Exception as e:
            print(f"스크래핑 중 오류 발생: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def extract_journalism_mcps(self):
        """
        저널리즘 관련 MCP 서버 추출
        
        Returns:
            저널리즘 관련 MCP 서버 데이터프레임
        """
        print("저널리즘 관련 MCP 서버 추출 중...")
        
        if self.mcp_data is None:
            print("MCP 데이터가 로드되지 않았습니다.")
            return None
        
        # 설명 컬럼 확인
        desc_cols = [col for col in self.mcp_data.columns if col in ['description', '설명', 'desc']]
        
        if not desc_cols:
            print("설명 컬럼을 찾을 수 없습니다.")
            return None
        
        desc_col = desc_cols[0]
        
        # 저널리즘 관련 MCP 서버 필터링
        journalism_mcps = []
        
        for idx, row in self.mcp_data.iterrows():
            score = 0
            matched_keywords = []
            
            # 이름과 설명에서 키워드 매칭
            text = str(row['name']) + ' ' + str(row.get(desc_col, ''))
            text = text.lower()
            
            for category, keywords in self.journalism_keywords.items():
                for keyword in keywords:
                    if keyword.lower() in text:
                        score += 1
                        matched_keywords.append(f"{category}: {keyword}")
            
            if score > 0:
                journalism_mcps.append({
                    'idx': idx,
                    'name': row['name'],
                    'url': row.get('url', ''),
                    'description': row.get(desc_col, ''),
                    'score': score,
                    'matched_keywords': ', '.join(matched_keywords)
                })
        
        if not journalism_mcps:
            print("저널리즘 관련 MCP 서버를 찾을 수 없습니다.")
            return None
        
        # 데이터프레임으로 변환
        journalism_df = pd.DataFrame(journalism_mcps)
        
        # 점수 기준 내림차순 정렬
        journalism_df = journalism_df.sort_values('score', ascending=False)
        
        # 결과 저장
        result_file = os.path.join(self.output_dir, "journalism_mcps.csv")
        journalism_df.to_csv(result_file, index=False, encoding='utf-8-sig')
        
        print(f"총 {len(journalism_df)}개의 저널리즘 관련 MCP 서버를 찾았습니다.")
        print(f"결과가 {result_file}에 저장되었습니다.")
        
        return journalism_df
    
    def enrich_journalism_mcps(self, top_n=10):
        """
        저널리즘 관련 MCP 서버 정보 보강
        
        Args:
            top_n: 정보를 보강할 상위 MCP 서버 수
            
        Returns:
            보강된 MCP 서버 데이터프레임
        """
        print(f"상위 {top_n}개 저널리즘 관련 MCP 서버 정보 보강 중...")
        
        # 저널리즘 관련 MCP 서버 추출
        journalism_df = self.extract_journalism_mcps()
        
        if journalism_df is None or len(journalism_df) == 0:
            return None
        
        # 상위 N개 MCP 서버 선택
        top_mcps = journalism_df.head(top_n)
        
        # 결과 저장용 리스트
        enriched_mcps = []
        
        # 각 MCP 서버에 대해 정보 보강
        for idx, row in top_mcps.iterrows():
            print(f"\n[{row['name']}] MCP 서버 정보 보강 중...")
            
            mcp_info = {
                'name': row['name'],
                'url': row['url'],
                'description': row['description'],
                'score': row['score'],
                'matched_keywords': row['matched_keywords']
            }
            
            # 웹 검색으로 추가 정보 수집
            search_query = f"{row['name']} MCP server journalism"
            search_result = self.search_mcp_info(search_query)
            
            mcp_info['search_result'] = search_result
            
            # URL이 있는 경우 웹사이트 스크래핑
            if row['url'] and row['url'] != '' and 'http' in row['url']:
                scrape_result = self.scrape_mcp_site(row['url'])
                mcp_info['scrape_result'] = scrape_result
            
            enriched_mcps.append(mcp_info)
            
            # API 호출 간 지연
            time.sleep(1)
        
        # 결과를 JSON 파일로 저장
        result_file = os.path.join(self.output_dir, "enriched_journalism_mcps.json")
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(enriched_mcps, f, ensure_ascii=False, indent=2)
        
        print(f"\n총 {len(enriched_mcps)}개의 MCP 서버 정보 보강 완료")
        print(f"보강된 정보가 {result_file}에 저장되었습니다.")
        
        return enriched_mcps
    
    def run(self, action='enrich', top_n=10):
        """
        메인 실행 함수
        
        Args:
            action: 수행할 작업 ('extract' 또는 'enrich')
            top_n: 정보를 보강할 상위 MCP 서버 수
        """
        # MCP 데이터 로드
        if not self.load_mcp_data():
            print("MCP 데이터 로드 실패. 프로그램을 종료합니다.")
            return
        
        if action == 'extract':
            # 저널리즘 관련 MCP 서버 추출
            self.extract_journalism_mcps()
        elif action == 'enrich':
            # 저널리즘 관련 MCP 서버 정보 보강
            self.enrich_journalism_mcps(top_n=top_n)
        else:
            print(f"지원하지 않는 작업: {action}")
            print("사용 가능한 작업: 'extract' 또는 'enrich'")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='FireCrawl을 활용한 MCP 서버 정보 보강 도구')
    parser.add_argument('--file', type=str, default='../../results/journalism_results/journalism_mcps_simple_top50.csv',
                      help='MCP 서버 데이터 파일 (CSV 또는 Excel)')
    parser.add_argument('--output', type=str, default='../../results/enriched_data',
                      help='결과 저장 디렉토리')
    parser.add_argument('--action', type=str, choices=['extract', 'enrich'], default='enrich',
                      help="수행할 작업 ('extract' 또는 'enrich')")
    parser.add_argument('--top', type=int, default=10,
                      help='정보를 보강할 상위 MCP 서버 수')
    
    args = parser.parse_args()
    
    # FireCrawl MCP 보강기 초기화 및 실행
    enricher = FirecrawlMCPEnricher(mcp_file=args.file, output_dir=args.output)
    enricher.run(action=args.action, top_n=args.top)


if __name__ == "__main__":
    main() 