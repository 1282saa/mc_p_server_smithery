"""
MCP 서버 API 테스트 도구

이 스크립트는 기자 업무에 유용한 MCP 서버의 API를 직접 호출하여 
실제 작동 여부와 유용성을 테스트하는 기능을 제공합니다.
"""

import requests
import json
import os
import sys
import time
import argparse
from datetime import datetime
import pandas as pd
from tqdm import tqdm

class MCPApiTester:
    """MCP 서버 API 테스트 클래스"""
    
    def __init__(self, mcp_list_file, output_dir='../../results/api_test_results'):
        """
        초기화 함수
        
        Args:
            mcp_list_file: 테스트할 MCP 서버 목록 파일 (CSV)
            output_dir: 결과 저장 디렉토리
        """
        self.mcp_list_file = mcp_list_file
        self.output_dir = output_dir
        self.mcp_list = None
        
        # 결과 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 테스트 쿼리 셋
        self.test_queries = {
            "뉴스 검색": [
                "최근 국내 주요 경제 뉴스",
                "전 세계 기후변화 관련 기사",
                "한국 반도체 산업 최신 동향"
            ],
            "웹 스크래핑": [
                "https://www.korea.kr/",
                "https://www.yna.co.kr/",
                "https://www.mk.co.kr/"
            ],
            "데이터 분석": [
                "한국 인구 통계 데이터 분석",
                "서울시 부동산 가격 추이 분석",
                "국내 백신 접종률 데이터 시각화"
            ]
        }
    
    def load_mcp_list(self):
        """MCP 서버 목록 로드"""
        print(f"MCP 서버 목록 파일 로드 중: {self.mcp_list_file}")
        
        try:
            self.mcp_list = pd.read_csv(self.mcp_list_file, encoding='utf-8-sig')
            print(f"총 {len(self.mcp_list)}개의 MCP 서버 정보 로드 완료")
            return True
        except Exception as e:
            print(f"MCP 서버 목록 로드 중 오류 발생: {str(e)}")
            return False
    
    def test_api(self, mcp_name, mcp_url, query_type, query):
        """
        MCP 서버 API 테스트
        
        Args:
            mcp_name: MCP 서버 이름
            mcp_url: MCP 서버 API URL
            query_type: 쿼리 유형
            query: 테스트 쿼리
            
        Returns:
            테스트 결과 사전
        """
        print(f"\n[{mcp_name}] '{query}' 테스트 중...")
        
        start_time = time.time()
        
        test_result = {
            "mcp_name": mcp_name,
            "mcp_url": mcp_url,
            "query_type": query_type,
            "query": query,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "error",
            "response_time": 0,
            "error_message": "",
            "response_data": None
        }
        
        try:
            # 요청 데이터 준비
            data = {"query": query}
            
            # API 요청
            response = requests.post(mcp_url, json=data, timeout=30)
            
            # 응답 시간 계산
            end_time = time.time()
            test_result["response_time"] = end_time - start_time
            
            # 응답 상태 확인
            if response.status_code == 200:
                test_result["status"] = "success"
                test_result["response_data"] = response.json()
                print(f"  - 성공 (응답 시간: {test_result['response_time']:.2f}초)")
            else:
                test_result["status"] = "error"
                test_result["error_message"] = f"HTTP 오류: {response.status_code}"
                print(f"  - 실패: {test_result['error_message']}")
        
        except requests.exceptions.RequestException as e:
            end_time = time.time()
            test_result["response_time"] = end_time - start_time
            test_result["error_message"] = f"요청 오류: {str(e)}"
            print(f"  - 실패: {test_result['error_message']}")
        
        except Exception as e:
            end_time = time.time()
            test_result["response_time"] = end_time - start_time
            test_result["error_message"] = f"일반 오류: {str(e)}"
            print(f"  - 실패: {test_result['error_message']}")
        
        return test_result
    
    def test_mcps(self, max_mcps=5, queries_per_mcp=2):
        """
        MCP 서버 API 테스트 실행
        
        Args:
            max_mcps: 테스트할 최대 MCP 서버 수
            queries_per_mcp: 각 MCP 서버당 테스트할 쿼리 수
            
        Returns:
            테스트 결과 목록
        """
        if self.mcp_list is None:
            if not self.load_mcp_list():
                return []
        
        # URL 컬럼 확인
        if 'url' not in self.mcp_list.columns:
            url_cols = [col for col in self.mcp_list.columns if 'url' in col.lower()]
            if url_cols:
                url_col = url_cols[0]
            else:
                print("URL 컬럼을 찾을 수 없습니다.")
                return []
        else:
            url_col = 'url'
        
        # 테스트할 MCP 서버 선택
        mcps_to_test = self.mcp_list.head(max_mcps)
        
        all_results = []
        
        # 각 MCP 서버 테스트
        for idx, row in mcps_to_test.iterrows():
            mcp_name = row.get('name', f"MCP-{idx}")
            mcp_url = row.get(url_col, "")
            
            if not mcp_url or 'http' not in mcp_url:
                print(f"[{mcp_name}] URL이 유효하지 않아 건너뜁니다: {mcp_url}")
                continue
            
            print(f"\n===== {mcp_name} 테스트 시작 =====")
            
            mcp_results = []
            
            # 각 쿼리 유형별 테스트
            for query_type, queries in self.test_queries.items():
                # 지정된 쿼리 수만큼만 테스트
                for query in queries[:queries_per_mcp]:
                    test_result = self.test_api(mcp_name, mcp_url, query_type, query)
                    mcp_results.append(test_result)
                    
                    # API 호출 간 지연
                    time.sleep(1)
            
            # 결과 저장
            all_results.extend(mcp_results)
            
            # MCP별 결과 저장
            mcp_result_file = os.path.join(self.output_dir, f"{mcp_name.replace(' ', '_')}_api_test.json")
            with open(mcp_result_file, 'w', encoding='utf-8') as f:
                json.dump(mcp_results, f, ensure_ascii=False, indent=2)
                
            print(f"{mcp_name} 테스트 결과 저장 완료: {mcp_result_file}")
        
        # 전체 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        all_result_file = os.path.join(self.output_dir, f"all_api_tests_{timestamp}.json")
        
        with open(all_result_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
            
        print(f"\n총 {len(all_results)}개 API 테스트 완료")
        print(f"전체 결과 저장 완료: {all_result_file}")
        
        # 결과 요약
        self.summarize_results(all_results)
        
        return all_results
    
    def summarize_results(self, results):
        """
        테스트 결과 요약
        
        Args:
            results: 테스트 결과 목록
        """
        if not results:
            print("요약할 결과가 없습니다.")
            return
        
        # MCP별 성공/실패 횟수
        mcp_stats = {}
        
        for result in results:
            mcp_name = result["mcp_name"]
            
            if mcp_name not in mcp_stats:
                mcp_stats[mcp_name] = {"success": 0, "error": 0, "total": 0, "avg_time": 0}
            
            mcp_stats[mcp_name]["total"] += 1
            
            if result["status"] == "success":
                mcp_stats[mcp_name]["success"] += 1
                mcp_stats[mcp_name]["avg_time"] += result["response_time"]
            else:
                mcp_stats[mcp_name]["error"] += 1
        
        # 평균 응답 시간 계산
        for mcp_name, stats in mcp_stats.items():
            if stats["success"] > 0:
                stats["avg_time"] = stats["avg_time"] / stats["success"]
        
        # 결과 출력
        print("\n===== API 테스트 결과 요약 =====")
        
        for mcp_name, stats in mcp_stats.items():
            success_rate = (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            print(f"\n[{mcp_name}]")
            print(f"  - 성공률: {success_rate:.1f}% ({stats['success']}/{stats['total']})")
            print(f"  - 평균 응답 시간: {stats['avg_time']:.2f}초")
        
        # 요약 결과 저장
        summary_file = os.path.join(self.output_dir, "api_test_summary.json")
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(mcp_stats, f, ensure_ascii=False, indent=2)
            
        print(f"\n요약 결과 저장 완료: {summary_file}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='MCP 서버 API 테스트 도구')
    parser.add_argument('--file', type=str, default='../../results/journalism_results/journalism_mcps_simple_top50.csv',
                      help='테스트할 MCP 서버 목록 파일 (CSV)')
    parser.add_argument('--output', type=str, default='../../results/api_test_results',
                      help='결과 저장 디렉토리')
    parser.add_argument('--max', type=int, default=5,
                      help='테스트할 최대 MCP 서버 수')
    parser.add_argument('--queries', type=int, default=2,
                      help='각 MCP 서버당 테스트할 쿼리 수')
    
    args = parser.parse_args()
    
    # MCP API 테스터 초기화 및 실행
    tester = MCPApiTester(mcp_list_file=args.file, output_dir=args.output)
    tester.test_mcps(max_mcps=args.max, queries_per_mcp=args.queries)


if __name__ == "__main__":
    main() 