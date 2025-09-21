#!/usr/bin/env python3
"""
SU:DA - 수화 인식 및 실시간 음성·텍스트 변환 시스템
메인 실행 진입점

사용법:
    python main.py --mode prepare    # 데이터 전처리
    python main.py --mode train      # 모델 학습
    python main.py --mode eval       # 모델 평가
    python main.py --mode infer      # 실시간 추론
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import traceback

from utils import (
    setup_logger, print_system_info, setup_directories,
    DATA_RAW_PATH, DATA_PROCESSED_PATH, CHECKPOINTS_PATH
)

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="SU:DA - 수화 인식 및 실시간 음성·텍스트 변환 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py --mode prepare                    # 데이터 전처리
  python main.py --mode train                      # 기본 설정으로 학습
  python main.py --mode train --epochs 50         # 50 에포크 학습
  python main.py --mode eval                       # 최고 모델로 평가
  python main.py --mode eval --checkpoint last.pt # 특정 체크포인트 평가
  python main.py --mode infer                      # 실시간 수어 인식
  python main.py --mode infer --camera 1          # 다른 카메라 사용

모드 설명:
  prepare: 원본 데이터를 학습용으로 전처리
  train:   BiLSTM 모델 학습
  eval:    학습된 모델 성능 평가
  infer:   실시간 웹캠 수어 인식
        """
    )
    
    # 필수 인수
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True,
        choices=['prepare', 'train', 'eval', 'infer'],
        help='실행 모드 선택'
    )
    
    # 공통 옵션
    parser.add_argument(
        '--vocab-path',
        type=Path,
        help='사전 파일 경로 (기본: data/processed/vocab/vocab.json)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help='학습/추론 디바이스 (기본: auto)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세 로그 출력'
    )
    
    # 학습 관련 옵션
    train_group = parser.add_argument_group('학습 옵션')
    train_group.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='학습 에포크 수 (기본: 100)'
    )
    
    train_group.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='배치 크기 (기본: 32)'
    )
    
    train_group.add_argument(
        '--learning-rate', '--lr',
        type=float,
        default=0.001,
        help='학습률 (기본: 0.001)'
    )
    
    train_group.add_argument(
        '--resume',
        type=Path,
        help='재개할 체크포인트 경로'
    )
    
    # 평가 관련 옵션
    eval_group = parser.add_argument_group('평가 옵션')
    eval_group.add_argument(
        '--checkpoint',
        type=Path,
        help='평가할 체크포인트 경로 (기본: best.pt)'
    )
    
    eval_group.add_argument(
        '--save-predictions',
        action='store_true',
        default=True,
        help='예측 결과 저장 (기본: True)'
    )
    
    eval_group.add_argument(
        '--quick-eval',
        type=int,
        help='빠른 평가 샘플 수 (전체 평가 건너뛰기)'
    )
    
    # 추론 관련 옵션
    infer_group = parser.add_argument_group('추론 옵션')
    infer_group.add_argument(
        '--camera',
        type=int,
        default=0,
        help='웹캠 ID (기본: 0)'
    )
    
    infer_group.add_argument(
        '--sequence-length',
        type=int,
        default=60,
        help='분석 시퀀스 길이 (기본: 60 프레임)'
    )
    
    infer_group.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.7,
        help='인식 신뢰도 임계값 (기본: 0.7)'
    )
    
    infer_group.add_argument(
        '--window-size',
        type=str,
        default='1280x720',
        help='윈도우 크기 (기본: 1280x720)'
    )
    
    return parser.parse_args()

def setup_device(device_arg: str):
    """디바이스 설정"""
    import torch
    
    if device_arg == 'auto':
        from utils import get_device
        return get_device()
    elif device_arg == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            print("⚠️  CUDA를 사용할 수 없습니다. CPU를 사용합니다.")
            return torch.device('cpu')
    elif device_arg == 'mps':
        if torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            print("⚠️  MPS를 사용할 수 없습니다. CPU를 사용합니다.")
            return torch.device('cpu')
    else:
        return torch.device('cpu')

def check_prerequisites(mode: str) -> bool:
    """전제 조건 확인"""
    logger = setup_logger("Prerequisites")
    
    if mode == 'prepare':
        # 원본 데이터 확인
        if not DATA_RAW_PATH.exists():
            logger.error(f"원본 데이터 경로가 없습니다: {DATA_RAW_PATH}")
            logger.error("preprocessing.py를 먼저 실행하여 데이터를 준비하세요.")
            return False
        
        required_dirs = ['keypoints', 'labels', 'videos']
        for dir_name in required_dirs:
            dir_path = DATA_RAW_PATH / dir_name
            if not dir_path.exists() or not any(dir_path.iterdir()):
                logger.error(f"필수 데이터 디렉토리가 비어있습니다: {dir_path}")
                return False
        
        logger.info("✅ 데이터 전처리 전제 조건 확인 완료")
    
    elif mode in ['train']:
        # 전처리된 데이터 확인
        if not DATA_PROCESSED_PATH.exists():
            logger.error(f"전처리된 데이터가 없습니다: {DATA_PROCESSED_PATH}")
            logger.error("먼저 'python main.py --mode prepare'를 실행하세요.")
            return False
        
        required_files = [
            DATA_PROCESSED_PATH / "splits" / "train.csv",
            DATA_PROCESSED_PATH / "splits" / "val.csv",
            DATA_PROCESSED_PATH / "vocab" / "vocab.json"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"필수 파일이 없습니다: {file_path}")
                logger.error("먼저 'python main.py --mode prepare'를 실행하세요.")
                return False
        
        logger.info("✅ 학습 전제 조건 확인 완료")
    
    elif mode in ['eval', 'infer']:
        # 학습된 모델 확인
        if not CHECKPOINTS_PATH.exists():
            logger.error(f"체크포인트 디렉토리가 없습니다: {CHECKPOINTS_PATH}")
            logger.error("먼저 모델을 학습하세요.")
            return False
        
        best_checkpoint = CHECKPOINTS_PATH / "best.pt"
        last_checkpoint = CHECKPOINTS_PATH / "last.pt"
        
        if not best_checkpoint.exists() and not last_checkpoint.exists():
            logger.error("학습된 모델이 없습니다.")
            logger.error("먼저 'python main.py --mode train'을 실행하세요.")
            return False
        
        logger.info("✅ 모델 사용 전제 조건 확인 완료")
    
    return True

def run_data_preparation(args) -> bool:
    """데이터 전처리 실행"""
    try:
        from data_preparation import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        success = preprocessor.run_preprocessing()
        
        if success:
            print("\n🎉 데이터 전처리가 성공적으로 완료되었습니다!")
            print(f"📂 전처리된 데이터 위치: {DATA_PROCESSED_PATH}")
            print("💡 다음 단계: python main.py --mode train")
        
        return success
        
    except Exception as e:
        print(f"❌ 데이터 전처리 실패: {e}")
        if args.verbose:
            traceback.print_exc()
        return False

def run_training(args) -> bool:
    """모델 학습 실행"""
    try:
        from train import SignLanguageTrainer
        
        device = setup_device(args.device)
        
        trainer = SignLanguageTrainer(
            vocab_path=args.vocab_path,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            device=device,
            resume_from=args.resume,
            use_class_weights=True,
            gradient_clip_val=1.0,
            save_every=5
        )
        
        trainer.train()
        
        print("\n🎉 모델 학습이 성공적으로 완료되었습니다!")
        print(f"📂 체크포인트 위치: {CHECKPOINTS_PATH}")
        print("💡 다음 단계: python main.py --mode eval")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 학습 실패: {e}")
        if args.verbose:
            traceback.print_exc()
        return False

def run_evaluation(args) -> bool:
    """모델 평가 실행"""
    try:
        from eval import SignLanguageEvaluator
        
        device = setup_device(args.device)
        
        evaluator = SignLanguageEvaluator(
            checkpoint_path=args.checkpoint,
            vocab_path=args.vocab_path,
            batch_size=args.batch_size,
            device=device,
            save_predictions=args.save_predictions
        )
        
        if args.quick_eval:
            # 빠른 평가
            results = evaluator.quick_evaluation(num_samples=args.quick_eval)
            print(f"\n⚡ 빠른 평가 완료 ({args.quick_eval}개 샘플)")
            print(f"정확도: {results['accuracy']:.2f}%")
        else:
            # 전체 평가
            results = evaluator.run_full_evaluation()
            print(f"\n🎉 모델 평가가 성공적으로 완료되었습니다!")
            print(f"검증 정확도: {results['val_metrics']['accuracy']:.2f}%")
        
        print("💡 다음 단계: python main.py --mode infer")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 평가 실패: {e}")
        if args.verbose:
            traceback.print_exc()
        return False

def run_inference(args) -> bool:
    """실시간 추론 실행"""
    try:
        from inference import SignLanguageInference
        
        device = setup_device(args.device)
        
        # 윈도우 크기 파싱
        try:
            width, height = map(int, args.window_size.split('x'))
            window_size = (width, height)
        except:
            print(f"⚠️  잘못된 윈도우 크기: {args.window_size}. 기본값 사용.")
            window_size = (1280, 720)
        
        inference_system = SignLanguageInference(
            checkpoint_path=args.checkpoint,
            vocab_path=args.vocab_path,
            sequence_length=args.sequence_length,
            confidence_threshold=args.confidence_threshold,
            detection_cooldown=1.5,
            device=device
        )
        
        print("\n🎥 실시간 수어 인식을 시작합니다!")
        print("💡 조작법: 'q'=종료, 'r'=초기화, 's'=로그저장")
        print("=" * 60)
        
        inference_system.run_webcam_inference(
            camera_id=args.camera,
            window_size=window_size
        )
        
        return True
        
    except Exception as e:
        print(f"❌ 실시간 추론 실패: {e}")
        if args.verbose:
            traceback.print_exc()
        return False

def main():
    """메인 함수"""
    # 인수 파싱
    args = parse_arguments()
    
    # 시스템 정보 출력
    if args.verbose:
        print_system_info()
    
    # 디렉토리 설정
    setup_directories()
    
    # 로거 설정
    logger = setup_logger("Main", "main.log")
    
    logger.info("=" * 80)
    logger.info(f"🚀 SU:DA 시스템 시작 - 모드: {args.mode}")
    logger.info("=" * 80)
    
    try:
        # 전제 조건 확인
        if not check_prerequisites(args.mode):
            logger.error("전제 조건 확인 실패")
            return False
        
        # 모드별 실행
        success = False
        
        if args.mode == 'prepare':
            success = run_data_preparation(args)
        
        elif args.mode == 'train':
            success = run_training(args)
        
        elif args.mode == 'eval':
            success = run_evaluation(args)
        
        elif args.mode == 'infer':
            success = run_inference(args)
        
        else:
            logger.error(f"알 수 없는 모드: {args.mode}")
            return False
        
        if success:
            logger.info(f"✅ {args.mode} 모드 실행 성공")
        else:
            logger.error(f"❌ {args.mode} 모드 실행 실패")
        
        return success
        
    except KeyboardInterrupt:
        logger.info("사용자에 의한 중단")
        print("\n👋 프로그램이 중단되었습니다.")
        return False
    
    except Exception as e:
        logger.error(f"예상치 못한 오류: {e}")
        if args.verbose:
            traceback.print_exc()
        return False

if __name__ == "__main__":
    import os
    
    # 현재 디렉토리가 프로젝트 루트인지 확인
    if not (Path.cwd() / "src").exists():
        print("❌ 프로젝트 루트 디렉토리에서 실행해주세요.")
        print("💡 올바른 실행 위치: Sign2Lang/")
        print("💡 현재 위치:", Path.cwd())
        sys.exit(1)
    
    # PATH에 src 추가
    src_path = str(Path.cwd() / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # 메인 실행
    success = main()
    exit_code = 0 if success else 1
    sys.exit(exit_code)