#!/usr/bin/env python3
"""
SU:DA - 수화 인식 및 실시간 음성·텍스트 변환 시스템
패키지 초기화 파일

이 패키지는 웹캠 입력을 통한 실시간 수어 인식 시스템을 제공합니다.
MediaPipe와 BiLSTM 모델을 사용하여 한국 수어를 텍스트로 변환합니다.
"""

__version__ = "1.0.0"
__author__ = "SU:DA Development Team"
__description__ = "수화 인식 및 실시간 음성·텍스트 변환 시스템"
__url__ = "https://github.com/suda-sign-language"

# 패키지 정보
PACKAGE_INFO = {
    "name": "SU:DA",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "url": __url__,
    "python_requires": ">=3.8",
    "keywords": ["sign-language", "computer-vision", "deep-learning", "mediapipe", "lstm"],
    "license": "MIT"
}

# 데이터 설정
DATA_CONFIG = {
    "word_range": (1501, 3000),
    "total_words": 1500,
    "speaker_id": "REAL01",
    "direction": "F",
    "sequence_length": 200,
    "keypoint_dim": 1629  # 543 landmarks * 3 (x, y, confidence)
}

# 모델 설정
MODEL_CONFIG = {
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "use_attention": True,
    "use_positional_encoding": False
}

# 학습 설정
TRAINING_CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "num_epochs": 100,
    "gradient_clip_val": 1.0,
    "save_every": 5
}

# 추론 설정
INFERENCE_CONFIG = {
    "sequence_length": 60,
    "confidence_threshold": 0.7,
    "detection_cooldown": 1.5,
    "camera_id": 0,
    "window_size": (1280, 720)
}

# 메인 모듈 임포트
try:
    from .utils import (
        setup_logger, setup_directories, get_device, print_system_info,
        DATA_RAW_PATH, DATA_PROCESSED_PATH, CHECKPOINTS_PATH, LOGS_PATH
    )
    
    from .vocab import SignVocabulary
    from .model import create_model, SignLanguageModel, BiLSTMSignClassifier
    from .dataset import SignLanguageDataset, create_data_loaders
    from .data_preparation import DataPreprocessor
    from .train import SignLanguageTrainer
    from .eval import SignLanguageEvaluator
    from .inference import SignLanguageInference, MediaPipeExtractor
    
    # 사용 가능한 모든 공개 클래스/함수
    __all__ = [
        # 유틸리티
        'setup_logger', 'setup_directories', 'get_device', 'print_system_info',
        'DATA_RAW_PATH', 'DATA_PROCESSED_PATH', 'CHECKPOINTS_PATH', 'LOGS_PATH',
        
        # 사전
        'SignVocabulary',
        
        # 모델
        'create_model', 'SignLanguageModel', 'BiLSTMSignClassifier',
        
        # 데이터셋
        'SignLanguageDataset', 'create_data_loaders',
        
        # 전처리
        'DataPreprocessor',
        
        # 학습
        'SignLanguageTrainer',
        
        # 평가
        'SignLanguageEvaluator',
        
        # 추론
        'SignLanguageInference', 'MediaPipeExtractor',
        
        # 설정
        'PACKAGE_INFO', 'DATA_CONFIG', 'MODEL_CONFIG', 
        'TRAINING_CONFIG', 'INFERENCE_CONFIG'
    ]

except ImportError as e:
    # 의존성이 없는 경우 경고 출력
    import warnings
    warnings.warn(f"일부 모듈을 임포트할 수 없습니다: {e}")
    
    # 기본적인 것들만 제공
    __all__ = [
        'PACKAGE_INFO', 'DATA_CONFIG', 'MODEL_CONFIG', 
        'TRAINING_CONFIG', 'INFERENCE_CONFIG'
    ]

def get_package_info():
    """패키지 정보 반환"""
    return PACKAGE_INFO.copy()

def get_data_config():
    """데이터 설정 반환"""
    return DATA_CONFIG.copy()

def get_model_config():
    """모델 설정 반환"""
    return MODEL_CONFIG.copy()

def get_training_config():
    """학습 설정 반환"""
    return TRAINING_CONFIG.copy()

def get_inference_config():
    """추론 설정 반환"""
    return INFERENCE_CONFIG.copy()

def print_package_info():
    """패키지 정보 출력"""
    print("=" * 60)
    print(f"📦 {PACKAGE_INFO['name']} v{PACKAGE_INFO['version']}")
    print("=" * 60)
    print(f"📝 설명: {PACKAGE_INFO['description']}")
    print(f"👨‍💻 개발자: {PACKAGE_INFO['author']}")
    print(f"📊 데이터: {DATA_CONFIG['total_words']}개 수어 단어")
    print(f"🎯 범위: WORD{DATA_CONFIG['word_range'][0]}~{DATA_CONFIG['word_range'][1]}")
    print(f"🧠 모델: BiLSTM (숨김층 {MODEL_CONFIG['hidden_dim']}, 레이어 {MODEL_CONFIG['num_layers']})")
    print(f"📹 실시간: {INFERENCE_CONFIG['sequence_length']}프레임 시퀀스 분석")
    print(f"🎯 신뢰도: {INFERENCE_CONFIG['confidence_threshold']*100}% 이상에서 인식")
    print("=" * 60)

def check_dependencies():
    """의존성 체크"""
    missing_deps = []
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import mediapipe
        print(f"✅ MediaPipe {mediapipe.__version__}")
    except ImportError:
        missing_deps.append("mediapipe")
    
    try:
        import numpy
        print(f"✅ NumPy {numpy.__version__}")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import pandas
        print(f"✅ Pandas {pandas.__version__}")
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError:
        missing_deps.append("scikit-learn")
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import seaborn
        print(f"✅ Seaborn {seaborn.__version__}")
    except ImportError:
        missing_deps.append("seaborn")
    
    try:
        import tqdm
        print(f"✅ TQDM {tqdm.__version__}")
    except ImportError:
        missing_deps.append("tqdm")
    
    if missing_deps:
        print(f"\n❌ 누락된 의존성: {', '.join(missing_deps)}")
        print("설치 명령어:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    else:
        print(f"\n✅ 모든 의존성이 설치되어 있습니다!")
        return True

# 개발 모드 체크
def is_development_mode():
    """개발 모드 여부 확인"""
    import os
    return os.environ.get('SUDA_DEV_MODE', 'false').lower() == 'true'

# 로깅 레벨 설정
def set_logging_level(level='INFO'):
    """로깅 레벨 설정"""
    import logging
    logging.getLogger('SU:DA').setLevel(getattr(logging, level.upper()))

# 패키지 초기화 시 실행
if __name__ != "__main__":
    # 개발 모드에서만 상세 정보 출력
    if is_development_mode():
        print_package_info()
        check_dependencies()

def main():
    """패키지 정보 확인용 메인 함수"""
    print_package_info()
    print("\n🔍 의존성 체크:")
    check_dependencies()
    
    print(f"\n📚 사용 가능한 모듈: {len(__all__)}개")
    print("주요 클래스:")
    print("  - SignLanguageInference: 실시간 수어 인식")
    print("  - SignLanguageTrainer: 모델 학습")
    print("  - SignLanguageEvaluator: 모델 평가")
    print("  - DataPreprocessor: 데이터 전처리")
    print("  - SignVocabulary: 수어 단어 사전")
    
    print(f"\n🎯 사용법:")
    print("  from src import SignLanguageInference")
    print("  inference = SignLanguageInference()")
    print("  inference.run_webcam_inference()")

if __name__ == "__main__":
    main()