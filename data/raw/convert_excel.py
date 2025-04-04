import pandas as pd

# 엑셀 파일 읽기
df = pd.read_excel('smithery_mcp_server.xlsx')

# CSV 파일로 저장
df.to_csv('smithery_mcp_server.csv', index=False, encoding='utf-8')

# 데이터프레임 정보 출력
print("\n=== 데이터프레임 정보 ===")
print(f"총 행 수: {len(df)}")
print(f"컬럼: {', '.join(df.columns)}")

# 처음 5개 행 출력
print("\n=== 처음 5개 행 ===")
print(df.head().to_string()) 