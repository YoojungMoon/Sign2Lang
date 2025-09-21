#!/usr/bin/env python3
"""
SU:DA - 수화 인식 시스템 유틸리티 함수들
공통으로 사용되는 함수들과 설정을 관리
"""

import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
from datetime import datetime

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUTS_PATH = PROJECT_ROOT / "outputs"
CHECKPOINTS_PATH = OUTPUTS_PATH / "checkpoints"
LOGS_PATH = OUTPUTS_PATH / "logs"

# 데이터 설정
WORD_START = 1501
WORD_END = 3000
TOTAL_WORDS = WORD_END - WORD_START + 1  # 1500개
SPEAKER_ID = "REAL01"
DIRECTION = "F"

# 모델 설정
KEYPOINT_DIM = 543  # MediaPipe: 33(pose) + 21*2(hands) + 468(face) = 543 landmarks
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 100

# 키포인트 구조 정의
POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
FACE_LANDMARKS = 468
TOTAL_LANDMARKS = POSE_LANDMARKS + HAND_LANDMARKS * 2 + FACE_LANDMARKS

def setup_directories():
    """필요한 디렉토리들을 생성"""
    directories = [
        DATA_PROCESSED_PATH / "sequences",
        DATA_PROCESSED_PATH / "splits", 
        DATA_PROCESSED_PATH / "vocab",
        CHECKPOINTS_PATH,
        LOGS_PATH,
        LOGS_PATH / "tensorboard"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    return directories

def setup_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """로거 설정"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 기존 핸들러 제거 (중복 방지)
    if logger.handlers:
        logger.handlers.clear()
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택적)
    if log_file:
        LOGS_PATH.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOGS_PATH / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_json(file_path: Path) -> Dict[str, Any]:
    """JSON 파일 로딩"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 파싱 오류 ({file_path}): {e}")

def save_json(data: Dict[str, Any], file_path: Path):
    """JSON 파일 저장"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_video_fps(video_path: Path) -> float:
    """비디오 파일의 FPS 추출"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if fps <= 0:
        fps = 30.0  # 기본값
        print(f"⚠️ FPS를 추출할 수 없어 기본값 {fps} 사용: {video_path}")
    
    return fps

def time_to_frame_index(time_seconds: float, fps: float) -> int:
    """시간(초)을 프레임 인덱스로 변환"""
    return int(time_seconds * fps)

def extract_keypoints_from_json(keypoint_json: Dict[str, Any]) -> np.ndarray:
    """키포인트 JSON에서 좌표 배열 추출"""
    try:
        people = keypoint_json.get("people", {})
        if isinstance(people, list) and len(people) > 0:
            person = people[0]
        elif isinstance(people, dict):
            person = people
        else:
            # 빈 키포인트 반환
            return np.zeros(TOTAL_LANDMARKS * 3)
        
        keypoints = []
        
        # Pose keypoints (33개)
        pose_2d = person.get("pose_keypoints_2d", [])
        if len(pose_2d) >= POSE_LANDMARKS * 3:
            keypoints.extend(pose_2d[:POSE_LANDMARKS * 3])
        else:
            keypoints.extend([0.0] * (POSE_LANDMARKS * 3))
        
        # Left hand keypoints (21개)
        hand_left_2d = person.get("hand_left_keypoints_2d", [])
        if len(hand_left_2d) >= HAND_LANDMARKS * 3:
            keypoints.extend(hand_left_2d[:HAND_LANDMARKS * 3])
        else:
            keypoints.extend([0.0] * (HAND_LANDMARKS * 3))
        
        # Right hand keypoints (21개)
        hand_right_2d = person.get("hand_right_keypoints_2d", [])
        if len(hand_right_2d) >= HAND_LANDMARKS * 3:
            keypoints.extend(hand_right_2d[:HAND_LANDMARKS * 3])
        else:
            keypoints.extend([0.0] * (HAND_LANDMARKS * 3))
        
        # Face keypoints (468개)
        face_2d = person.get("face_keypoints_2d", [])
        if len(face_2d) >= FACE_LANDMARKS * 3:
            keypoints.extend(face_2d[:FACE_LANDMARKS * 3])
        else:
            keypoints.extend([0.0] * (FACE_LANDMARKS * 3))
        
        return np.array(keypoints, dtype=np.float32)
        
    except Exception as e:
        print(f"⚠️ 키포인트 추출 오류: {e}")
        return np.zeros(TOTAL_LANDMARKS * 3, dtype=np.float32)

def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """키포인트 정규화 (x, y는 0-1로, confidence는 그대로)"""
    if len(keypoints) == 0:
        return keypoints
    
    # 3개씩 묶어서 처리 (x, y, confidence)
    normalized = keypoints.copy()
    for i in range(0, len(keypoints), 3):
        if i + 2 < len(keypoints):
            # x, y 좌표만 정규화 (0-1 범위로 가정)
            normalized[i] = np.clip(keypoints[i], 0, 1)      # x
            normalized[i+1] = np.clip(keypoints[i+1], 0, 1)  # y
            # confidence는 그대로 유지
            normalized[i+2] = np.clip(keypoints[i+2], 0, 1)  # conf
    
    return normalized

def pad_sequence(sequence: np.ndarray, max_length: int, pad_value: float = 0.0) -> np.ndarray:
    """시퀀스를 고정 길이로 패딩"""
    seq_len = len(sequence)
    
    if seq_len >= max_length:
        # 길이가 긴 경우 앞부분 자르기
        return sequence[:max_length]
    else:
        # 길이가 짧은 경우 뒤에 패딩 추가
        pad_length = max_length - seq_len
        if len(sequence.shape) == 1:
            padding = np.full(pad_length, pad_value)
        else:
            padding = np.full((pad_length, sequence.shape[1]), pad_value)
        return np.concatenate([sequence, padding], axis=0)

def save_checkpoint(model, optimizer, epoch: int, loss: float, accuracy: float, 
                   checkpoint_path: Path, is_best: bool = False):
    """모델 체크포인트 저장"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }
    
    # 기본 체크포인트 저장
    torch.save(checkpoint, checkpoint_path)
    
    # 최고 성능 모델 저장
    if is_best:
        best_path = checkpoint_path.parent / "best.pt"
        torch.save(checkpoint, best_path)
        print(f"✅ 최고 성능 모델 저장: {best_path}")

def load_checkpoint(checkpoint_path: Path, model, optimizer=None) -> Dict[str, Any]:
    """모델 체크포인트 로딩"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"체크포인트 파일이 없습니다: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 모델 가중치 로딩
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 옵티마이저 가중치 로딩 (선택적)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✅ 체크포인트 로딩 완료: epoch {checkpoint['epoch']}, "
          f"loss {checkpoint['loss']:.4f}, accuracy {checkpoint['accuracy']:.4f}")
    
    return checkpoint

def get_device() -> torch.device:
    """사용 가능한 디바이스 반환"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA 사용: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Apple MPS 사용")
    else:
        device = torch.device("cpu")
        print("✅ CPU 사용")
    
    return device

def count_parameters(model) -> int:
    """모델 파라미터 개수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_word_id_from_filename(filename: str) -> int:
    """파일명에서 단어 ID 추출 (예: NIA_SL_WORD1501_REAL01_F -> 1501)"""
    try:
        parts = filename.split('_')
        for part in parts:
            if part.startswith('WORD'):
                return int(part[4:])  # 'WORD' 제거 후 숫자 추출
        raise ValueError(f"파일명에서 단어 ID를 찾을 수 없습니다: {filename}")
    except (ValueError, IndexError):
        raise ValueError(f"잘못된 파일명 형식: {filename}")

def is_valid_word_id(word_id: int) -> bool:
    """유효한 단어 ID 범위 체크"""
    return WORD_START <= word_id <= WORD_END

def create_word_range() -> List[int]:
    """학습에 사용할 단어 ID 범위 생성"""
    return list(range(WORD_START, WORD_END + 1))

def format_time(seconds: float) -> str:
    """초를 시:분:초 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def print_system_info():
    """시스템 정보 출력"""
    print("=" * 50)
    print("🚀 SU:DA - 수화 인식 시스템")
    print("=" * 50)
    print(f"📂 프로젝트 루트: {PROJECT_ROOT}")
    print(f"📊 데이터 범위: WORD{WORD_START}~{WORD_END} ({TOTAL_WORDS}개)")
    print(f"👤 화자: {SPEAKER_ID}")
    print(f"📹 방향: {DIRECTION} (정면)")
    print(f"🎯 키포인트 차원: {TOTAL_LANDMARKS * 3}")
    print(f"💾 디바이스: {get_device()}")
    print("=" * 50)

if __name__ == "__main__":
    # 테스트용 코드
    print_system_info()
    setup_directories()
    logger = setup_logger("utils_test", "test.log")
    logger.info("유틸리티 모듈 테스트 완료")