# MCP 서버 분석 프로젝트

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

기자 및 신문사를 위한 MCP(Model Context Protocol) 서버 분석 및 분류 도구입니다.

## 프로젝트 소개

이 프로젝트는 기자 및 신문사에서 활용할 수 있는 MCP 서버를 찾고 분석하기 위한 다양한 도구를 제공합니다. MCP 서버는 AI가 웹에서 자동으로 정보를 찾고 처리할 수 있게 하는 표준화된 방식을 제공하는 서비스로, 기자 업무에 큰 도움이 될 수 있습니다.

### 주요 기능

- MCP 서버 데이터 수집 및 분석
- 기자 업무 관련 MCP 서버 식별 및 추천
- 다양한 분석 방법론 (키워드 기반, 텍스트 임베딩, 유사도 분석 등)
- 웹 검색 및 스크래핑을 통한 MCP 서버 정보 보강
- MCP 서버 API 테스트 및 성능 평가
- 기자 워크플로우에 MCP 서버 통합 가이드

## 설치 방법

```bash
# 저장소 복제
git clone https://github.com/1282saa/mc_p_server_smithery.git
cd mc_p_server_smithery

# 필요한 패키지 설치
pip install -r requirements.txt
```

## 프로젝트 구조

```
mc_p_server_smithery/
├── data/                      # 모든 데이터 파일
│   ├── raw/                   # 원본 데이터 파일
│   └── processed/             # 처리된 데이터 파일
│
├── scripts/                   # 모든 Python 스크립트
│   ├── basic/                 # 기본 데이터 처리 스크립트
│   ├── analysis/              # 분석용 스크립트
│   └── utils/                 # 유틸리티 스크립트
│
├── notebooks/                 # Jupyter 노트북 파일들
│
├── results/                   # 모든 분석 결과
│   ├── news_mcp_results/
│   ├── topic_modeling_results/
│   └── other_results/
│
└── docs/                      # 문서 파일들
    └── guides/                # 사용 가이드 등
```

## 주요 스크립트 사용법

### 기자 업무 분석 도구

```bash
# 통합 기자 분석 도구 실행
python scripts/analysis/journalism_analysis.py --mode advanced

# 결과 확인
ls results/journalism_results/
```

### MCP 서버 정보 보강 도구

```bash
# FireCrawl을 활용한 MCP 서버 정보 보강
python scripts/utils/firecrawl_mcp_enricher.py --action enrich --top 10

# 결과 확인
ls results/enriched_data/
```

### API 테스트 도구

```bash
# MCP 서버 API 테스트
python scripts/utils/mcp_api_tester.py --max 5 --queries 2

# 결과 확인
ls results/api_test_results/
```

## 주요 분석 결과

- [기자 업무별 추천 MCP 서버](docs/guides/journalist_workflow_guide.md)
- [MCP 서버 분석 요약](docs/SUMMARY.md)

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 기여 방법

이슈 등록이나 풀 리퀘스트를 통해 이 프로젝트에 기여할 수 있습니다. 기여하기 전에 프로젝트의 코드 스타일과 기여 가이드라인을 확인해주세요.
