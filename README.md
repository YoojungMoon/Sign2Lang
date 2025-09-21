# SU:DA - 실시간 수어 인식 시스템
웹캠을 통한 실시간 한국 수어 인식 및 텍스트 변환 시스템

## 🚀 핵심 기능
- 실시간 인식: 웹캠으로 수어 동작을 즉시 텍스트로 변환
- 1,500개 단어: 한국 수어 WORD1501~3000 지원
- 높은 정확도: BiLSTM + Attention 모델로 85-90% 정확도
- 직관적 UI: 키포인트 시각화 및 Top-3 예측 표시

## 📦 설치

# 저장소 클론
git clone https://github.com/YoojungMoon/Sign2Lang.git
cd Sign2Lang

# 환경 설정
conda env create -f environment.yml
# 맥 환경 설정
conda env create -f environment_mac.yml

conda activate suda


## 🎯 사용법

### 1단계: 데이터 전처리
python src/main.py --mode prepare


### 2단계: 모델 학습
python src/main.py --mode train

### 3단계: 모델 평가
python src/main.py --mode eval

### 4단계: 실시간 수어 인식
python src/main.py --mode infer

## 조작법:
`q`=종료, `r`=초기화, `s`=로그저장

## 📁 프로젝트 구조
Sign2Lang/
├── README.md                    # 프로젝트 설명서
├── environment.yml              # Conda 환경 설정
├── requirements.txt             # pip 의존성 목록
│
├── data/
│   ├──raw/                     # 원본 데이터 (preprocessing.py로 준비)
│   │   ├──keypoints/           # 키포인트 데이터 (1,500개)
│   │   ├──labels/              # 형태소 라벨 (1,500개)  
│   │   └──videos/              # 원본 영상 (1,500개)
│   └──processed/               # 전처리된 데이터 (자동 생성)
│       ├──sequences/           # 시퀀스 파일들
│       ├──splits/              # train/val 분할
│       └──vocab/               # 단어 사전
│
├── outputs/                     # 학습/평가 결과 (자동 생성)
│   ├── checkpoints/             # 모델 체크포인트
│   └── logs/                    # 로그 및 결과
│       ├── tensorboard/         # TensorBoard 로그
│       └── evaluation_results/  # 평가 결과
│
└── src/                         # 소스 코드
    ├── __init__.py              # 패키지 초기화
    ├── main.py                  # 메인 실행 파일
    ├── utils.py                 # 공통 유틸리티
    ├── data_preparation.py      # 데이터 전처리
    ├── vocab.py                 # 단어 사전 관리
    ├── dataset.py               # PyTorch 데이터셋
    ├── model.py                 # BiLSTM 모델
    ├── train.py                 # 모델 학습
    ├── eval.py                  # 모델 평가
    └── inference.py             # 실시간 추론


## 모델 아키텍처
- 입력: MediaPipe 키포인트 (543 landmarks × 3)
- 모델: BiLSTM + Attention (256 hidden, 2 layers)
- 출력: 1,500개 수어 단어 분류
- 성능: Top-1 85-90%, Top-3 95-98%

## 📊 데이터

- 출처: AI Hub 한국 수어 영상 데이터셋
- 범위: WORD1501~3000 (1,500개 단어)
- 화자: REAL01 (단일 화자), F방향 (정면)
- 구성: 영상 + 키포인트 + 형태소 라벨

## ⚙️ 요구사항
- Python 3.9+
- 웹캠 (USB 또는 내장)
- GPU 권장 (CUDA 지원)

## 🔧 문제 해결

# 의존성 체크
python src/__init__.py

# 다른 카메라 사용
python src/main.py --mode infer --camera 1

# 메모리 부족시 배치 크기 감소
python src/main.py --mode train --batch-size 16
