import subprocess
import os
import time
import sys

# 필요한 패키지 설치 여부 확인 및 설치 함수
def check_and_install_packages():
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn',
        'wordcloud', 'umap-learn', 'openpyxl'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 패키지가 이미 설치되어 있습니다.")
        except ImportError:
            print(f"✗ {package} 패키지를 설치합니다...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} 패키지 설치 완료!")

# 스크립트 실행 함수
def run_script(script_path, description):
    print(f"\n\n{'=' * 50}")
    print(f"  {description} 시작")
    print(f"{'=' * 50}\n")
    
    start_time = time.time()
    
    try:
        process = subprocess.Popen([sys.executable, script_path], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True)
        
        # 실시간 출력 처리
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # 에러 확인
        stderr = process.stderr.read()
        if stderr:
            print(f"오류 발생:\n{stderr}")
        
        process.wait()
        
        if process.returncode == 0:
            elapsed_time = time.time() - start_time
            print(f"\n✓ {description} 완료! (소요 시간: {elapsed_time:.2f}초)")
            return True
        else:
            print(f"\n✗ {description} 실패! (종료 코드: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"\n✗ {description} 실행 중 오류 발생: {str(e)}")
        return False

# 결과 디렉토리 생성
def create_results_directory():
    results_dir = "classification_results"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

# 메인 함수
def main():
    print("\n" + "=" * 60)
    print(" " * 15 + "엑셀 시트 데이터 분류 도구")
    print("=" * 60)
    
    # 패키지 설치 확인
    print("\n[1/5] 필요한 패키지 설치 여부 확인 중...")
    check_and_install_packages()
    
    # 결과 디렉토리 생성
    print("\n[2/5] 결과 디렉토리 생성 중...")
    results_dir = create_results_directory()
    print(f"✓ 결과 디렉토리 생성 완료: {results_dir}")
    
    # 각 분류 스크립트 실행
    scripts = [
        {"path": "classify_data.py", "description": "K-means 클러스터링 기반 분류"},
        {"path": "topic_modeling.py", "description": "토픽 모델링(LDA) 기반 분류"},
        {"path": "similarity_clustering.py", "description": "유사도 기반 계층적 클러스터링"}
    ]
    
    print("\n[3/5] 분류 스크립트 실행 중...")
    success_count = 0
    for idx, script in enumerate(scripts, 1):
        if run_script(script["path"], script["description"]):
            success_count += 1
    
    # 결과 요약
    print("\n[4/5] 분류 결과 요약 중...")
    if success_count == len(scripts):
        print(f"✓ 모든 분류 작업이 성공적으로 완료되었습니다! ({success_count}/{len(scripts)})")
    else:
        print(f"! 일부 분류 작업이 실패했습니다. ({success_count}/{len(scripts)})")
    
    # 결과 통합 디렉토리로 복사
    print("\n[5/5] 분류 결과 통합 중...")
    try:
        # CSV 파일 목록 가져오기
        csv_files = []
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('_classified.csv') or file.endswith('_topics.csv') or file.endswith('_similarity_clusters.csv'):
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(results_dir, file)
                    # 파일 복사
                    with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
                        dst.write(src.read())
                    csv_files.append(dst_path)
        
        # 요약 보고서 작성
        with open(os.path.join(results_dir, '분류_결과_종합_요약.txt'), 'w', encoding='utf-8') as f:
            f.write("# 엑셀 시트 데이터 분류 종합 결과\n\n")
            f.write(f"총 {len(csv_files)}개의 분류 결과 파일이 생성되었습니다.\n\n")
            
            f.write("## 분류 방법별 결과 파일\n\n")
            
            # 분류 방법별 파일 목록
            kmeans_files = [f for f in csv_files if f.endswith('_classified.csv')]
            topic_files = [f for f in csv_files if f.endswith('_topics.csv')]
            similarity_files = [f for f in csv_files if f.endswith('_similarity_clusters.csv')]
            
            f.write(f"### K-means 클러스터링 결과 ({len(kmeans_files)}개 파일)\n")
            for file in kmeans_files:
                f.write(f"- {os.path.basename(file)}\n")
            f.write("\n")
            
            f.write(f"### 토픽 모델링(LDA) 결과 ({len(topic_files)}개 파일)\n")
            for file in topic_files:
                f.write(f"- {os.path.basename(file)}\n")
            f.write("\n")
            
            f.write(f"### 유사도 기반 클러스터링 결과 ({len(similarity_files)}개 파일)\n")
            for file in similarity_files:
                f.write(f"- {os.path.basename(file)}\n")
            f.write("\n")
            
            f.write("## 시트별 분류 결과\n\n")
            
            # 시트별 분류 결과
            sheet_names = set()
            for file in csv_files:
                base_name = os.path.basename(file)
                sheet_name = base_name.split('_')[0]
                if len(base_name.split('_')) > 1:
                    sheet_name = '_'.join(base_name.split('_')[:-1])
                sheet_names.add(sheet_name)
            
            for sheet in sorted(sheet_names):
                sheet_files = [f for f in csv_files if os.path.basename(f).startswith(sheet)]
                f.write(f"### {sheet} 시트 분류 결과 ({len(sheet_files)}개 파일)\n")
                for file in sheet_files:
                    f.write(f"- {os.path.basename(file)}\n")
                f.write("\n")
        
        print(f"✓ 분류 결과 통합 완료! 결과 요약: {os.path.join(results_dir, '분류_결과_종합_요약.txt')}")
        
    except Exception as e:
        print(f"! 결과 통합 중 오류 발생: {str(e)}")
    
    print("\n" + "=" * 60)
    print(" " * 15 + "분류 작업이 완료되었습니다!")
    print("=" * 60 + "\n")
    print(f"결과는 '{results_dir}' 디렉토리에서 확인할 수 있습니다.")

if __name__ == "__main__":
    main() 