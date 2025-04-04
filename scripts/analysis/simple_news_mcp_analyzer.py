import pandas as pd
import os

# 결과 저장 디렉토리 생성
output_dir = "news_mcp_results"
os.makedirs(output_dir, exist_ok=True)

# 엑셀 파일 경로
excel_file = 'smithery/smithery_mcp_server.xlsx'

print("엑셀 파일에서 데이터 로드 중...")
# 모든 시트 데이터 로드
try:
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
    print(f"총 {len(all_data)}개의 MCP 서버 데이터 통합 완료")
    
    # 기자/신문사 업무 관련 키워드 정의
    news_keywords = [
        # 취재 관련
        "news", "journalism", "reporter", "interview", "research", "investigation", "story", "article", 
        "media", "press", "newspaper", "broadcast", "publishing", "editorial", "editor", "뉴스", "기자", 
        "취재", "인터뷰", "조사", "기사", "미디어", "언론", "신문", "방송", "출판", "편집",
        
        # 데이터 분석 관련
        "data analysis", "statistics", "trends", "insights", "visualization", "차트", "그래프",
        "데이터 분석", "통계", "트렌드", "인사이트", "시각화",
        
        # 콘텐츠 생성 관련
        "content", "writing", "summarize", "summarization", "paraphrase", "translation", 
        "grammar", "proofreading", "edit", "headline", "title", "caption",
        "콘텐츠", "글쓰기", "요약", "번역", "문법", "교정", "편집", "헤드라인", "제목", "캡션",
        
        # 팩트체크 관련
        "fact check", "verification", "accuracy", "source", "reference", "authenticate", 
        "validate", "credibility", "팩트체크", "검증", "정확성", "출처", "참조", "신뢰성",
        
        # 소셜미디어 관련
        "social media", "twitter", "facebook", "instagram", "social network", "trending", 
        "viral", "소셜미디어", "트위터", "페이스북", "인스타그램", "소셜 네트워크", "트렌딩", "바이럴",
        
        # AI 관련
        "ai writing", "ai journalism", "automated journalism", "robot journalism", 
        "ai 글쓰기", "ai 저널리즘", "자동화 저널리즘", "로봇 저널리즘",
        
        # 웹 크롤링/스크래핑 관련
        "web crawling", "scraping", "data extraction", "web automation",
        "웹 크롤링", "스크래핑", "데이터 추출", "웹 자동화"
    ]
    
    # 기자 업무 시나리오 정의
    journalism_scenarios = {
        "기사 작성 보조": ["writing", "summarize", "grammar", "proofreading", "edit", "headline", "content", "글쓰기", "요약", "문법", "교정", "편집", "헤드라인", "제목", "콘텐츠"],
        "팩트 체크 및 검증": ["fact check", "verification", "accuracy", "source", "reference", "팩트체크", "검증", "정확성", "출처", "참조", "신뢰성"],
        "데이터 저널리즘": ["data analysis", "statistics", "visualization", "charts", "graphs", "데이터 분석", "통계", "시각화", "차트", "그래프"],
        "트렌드 분석": ["trends", "social media", "trending", "viral", "트렌드", "소셜미디어", "트렌딩", "바이럴"],
        "멀티미디어 콘텐츠": ["image", "audio", "video", "이미지", "오디오", "비디오", "멀티미디어"],
        "취재 보조": ["interview", "research", "investigation", "인터뷰", "조사", "취재"],
        "국제 뉴스 보도": ["translation", "international", "global", "번역", "국제", "글로벌"],
        "아카이브 및 검색": ["search", "archive", "retrieve", "find", "검색", "아카이브", "찾기"]
    }
    
    print("\n1. 기자/신문사 관련 MCP 서버 탐색 중...")
    
    # 각 설명에서 키워드 매칭 점수 계산
    all_data['relevance_score'] = 0
    for idx, row in all_data.iterrows():
        if pd.notna(row.get('설명')):
            description = str(row['설명']).lower()
            score = sum(1 for keyword in news_keywords if keyword.lower() in description)
            all_data.at[idx, 'relevance_score'] = score
    
    # 점수 기준 내림차순 정렬
    news_related_mcps = all_data.sort_values('relevance_score', ascending=False)
    
    # 상위 50개 관련 서버 저장
    top_news_related = news_related_mcps.head(50)
    top_news_related.to_csv(os.path.join(output_dir, "신문사_관련_MCP서버_TOP50.csv"), encoding='utf-8-sig', index=False)
    print(f"기자/신문사 관련 상위 50개 MCP 서버 목록 저장 완료: {os.path.join(output_dir, '신문사_관련_MCP서버_TOP50.csv')}")
    
    # 시트별 분포 계산
    sheet_counts = top_news_related['sheet_name'].value_counts()
    print("\n상위 50개 MCP 서버의 시트별 분포:")
    for sheet, count in sheet_counts.items():
        print(f"- {sheet}: {count}개")
    
    # 시나리오별 MCP 서버 찾기
    print("\n2. 기자 업무 시나리오별 MCP 서버 찾기...")
    
    for scenario, keywords in journalism_scenarios.items():
        print(f"\n[{scenario}] 시나리오 관련 MCP 서버:")
        
        # 각 설명에서 시나리오 키워드 매칭 점수 계산
        all_data[f'{scenario}_score'] = 0
        for idx, row in all_data.iterrows():
            if pd.notna(row.get('설명')):
                description = str(row['설명']).lower()
                score = sum(1 for keyword in keywords if keyword.lower() in description)
                all_data.at[idx, f'{scenario}_score'] = score
        
        # 점수 기준 내림차순 정렬 및 상위 10개 출력
        scenario_related = all_data.sort_values(f'{scenario}_score', ascending=False)
        top_scenario = scenario_related.head(10)
        
        # 결과 저장
        scenario_file = f"{scenario.replace(' ', '_')}_추천_MCP서버.csv"
        top_scenario.to_csv(os.path.join(output_dir, scenario_file), encoding='utf-8-sig', index=False)
        print(f"  추천 서버 저장 완료: {os.path.join(output_dir, scenario_file)}")
        
        # 상위 3개 서버 출력
        for i, (_, row) in enumerate(top_scenario.head(3).iterrows()):
            print(f"  {i+1}. {row.get('name', '')}: {row.get('설명', '')[:100]}...")
    
    # 종합 추천 결과 계산 (모든 시나리오 점수의 합)
    print("\n3. 종합 추천 MCP 서버 계산 중...")
    
    # 모든 시나리오 점수 합산
    scenario_columns = [f'{scenario}_score' for scenario in journalism_scenarios.keys()]
    all_data['total_scenario_score'] = all_data[scenario_columns].sum(axis=1)
    
    # 점수 기준 내림차순 정렬
    all_data_sorted = all_data.sort_values('total_scenario_score', ascending=False)
    
    # 상위 20개 종합 추천 서버 저장
    top_overall = all_data_sorted.head(20)
    top_overall.to_csv(os.path.join(output_dir, "종합_추천_MCP서버_TOP20.csv"), encoding='utf-8-sig', index=False)
    print(f"종합 추천 상위 20개 MCP 서버 목록 저장 완료: {os.path.join(output_dir, '종합_추천_MCP서버_TOP20.csv')}")
    
    # 상위 5개 서버 출력
    print("\n종합 추천 상위 5개 MCP 서버:")
    for i, (_, row) in enumerate(top_overall.head(5).iterrows()):
        print(f"{i+1}. {row.get('name', '')}: {row.get('설명', '')[:100]}...")
        print(f"   URL: {row.get('url', '')}")
        print(f"   카테고리: {row.get('sheet_name', '')}")
        print(f"   종합 점수: {row.get('total_scenario_score', 0)}")
        print("")
    
    print("\n분석이 완료되었습니다. 신문사 업무에 적합한 MCP 서버 추천 결과를 확인하세요.")
    
except Exception as e:
    print(f"오류 발생: {str(e)}")
    import traceback
    traceback.print_exc() 