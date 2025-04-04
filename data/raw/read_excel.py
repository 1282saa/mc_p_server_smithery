import pandas as pd
import sys

try:
    # 출력 옵션 설정
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    # 엑셀 파일 읽기
    excel_file = pd.ExcelFile('smithery_mcp_server.xlsx')

    # 시트 이름 출력
    print("\n=== 엑셀 파일의 시트 목록 ===")
    print(excel_file.sheet_names)

    # 각 시트별로 처리
    for sheet_name in excel_file.sheet_names:
        print(f"\n=== {sheet_name} 시트 ===")
        
        # 데이터 읽기
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # 실제 데이터가 있는 행부터 시작
        start_row = df[df['Unnamed: 0'].notna()].index[0]
        df = df.iloc[start_row:].reset_index(drop=True)
        
        # 컬럼 이름 정리
        df.columns = ['index', 'category', 'search_query', 'name', 'description', 'description_ko', 'url', 'type', 'usage_count']
        
        # 불필요한 컬럼 제거
        df = df.drop(['index', 'category'], axis=1)
        
        # NaN 값 제거
        df = df.dropna(how='all')
        
        # 처음 5개 행 출력
        print("\n처음 5개 행:")
        if len(df) > 0:
            print(df[['name', 'description_ko', 'type', 'usage_count']].head().to_string(index=False))
        else:
            print("데이터가 없습니다.")
        
        # 데이터프레임 정보 출력
        print(f"\n데이터프레임 정보:")
        print(f"총 행 수: {len(df)}")
        print(f"컬럼: {', '.join(df.columns)}")
        print("-" * 80)

except Exception as e:
    print(f"오류 발생: {str(e)}", file=sys.stderr) 