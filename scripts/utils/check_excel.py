"""
MCP 서버 엑셀 파일 구조 확인 스크립트
"""

import pandas as pd
import os
import sys

def check_excel_structure(excel_path):
    """엑셀 파일의 구조(시트 및 데이터 샘플) 확인"""
    try:
        print(f"엑셀 파일 경로: {excel_path}")
        
        # 엑셀 파일 로드
        excel_file = pd.ExcelFile(excel_path)
        
        # 시트 목록 출력
        print(f"\n엑셀 파일의 시트 목록:")
        for idx, sheet_name in enumerate(excel_file.sheet_names):
            print(f"{idx+1}. {sheet_name}")
        
        # 각 시트별 데이터 샘플 확인
        print("\n각 시트별 데이터 샘플:")
        for sheet_name in excel_file.sheet_names:
            try:
                # 첫 5개 행 읽기 (헤더 포함)
                df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=5)
                print(f"\n[{sheet_name}] 시트:")
                print(f"- 컬럼 수: {len(df.columns)}")
                print(f"- 컬럼 목록: {df.columns.tolist()}")
                print(f"- 데이터 샘플: {len(df)} 행")
            except Exception as e:
                print(f"시트 '{sheet_name}' 읽기 실패: {str(e)}")
        
        return True
    except Exception as e:
        print(f"엑셀 파일 확인 중 오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    # 기본 경로 설정
    default_path = os.path.join("..", "..", "data", "smithery_mcp_server.xlsx")
    
    # 명령행 인자로 경로를 받을 수 있음
    excel_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    # 엑셀 구조 확인
    check_excel_structure(excel_path) 