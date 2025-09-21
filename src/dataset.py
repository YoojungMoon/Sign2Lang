#!/usr/bin/env python3
"""
SU:DA - 수어 데이터셋 클래스
PyTorch Dataset 구현으로 학습/평가용 데이터 로딩
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import random

from utils import (
    setup_logger, DATA_PROCESSED_PATH, pad_sequence,
    BATCH_SIZE
)
from vocab import SignVocabulary

class SignLanguageDataset(Dataset):
    """수어 인식용 PyTorch Dataset"""
    
    def __init__(self, 
                 split: str = "train",
                 vocab_path: Optional[Path] = None,
                 max_sequence_length: int = 200,
                 augment: bool = False,
                 normalize: bool = True):
        """
        Args:
            split: 데이터 분할 ("train" 또는 "val")
            vocab_path: 사전 파일 경로
            max_sequence_length: 최대 시퀀스 길이
            augment: 데이터 증강 사용 여부
            normalize: 정규화 사용 여부
        """
        super().__init__()
        
        self.logger = setup_logger(f"SignDataset_{split}")
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.augment = augment and (split == "train")  # 학습시에만 증강
        self.normalize = normalize
        
        # 경로 설정
        self.processed_path = DATA_PROCESSED_PATH
        self.sequences_path = self.processed_path / "sequences"
        self.splits_path = self.processed_path / "splits"
        
        # 사전 로딩
        self.vocab = SignVocabulary(vocab_path)
        self.logger.info(f"사전 로딩 완료: {len(self.vocab)}개 단어")
        
        # 데이터 메타데이터 로딩
        self.metadata = self._load_metadata()
        self.logger.info(f"{split} 데이터셋 로딩 완료: {len(self.metadata)}개 샘플")
        
        # 통계 정보
        self._compute_statistics()
    
    def _load_metadata(self) -> pd.DataFrame:
        """분할된 데이터 메타데이터 로딩"""
        metadata_file = self.splits_path / f"{self.split}.csv"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"메타데이터 파일이 없습니다: {metadata_file}")
        
        try:
            metadata = pd.read_csv(metadata_file)
            self.logger.info(f"메타데이터 로딩: {len(metadata)}개 샘플")
            return metadata
            
        except Exception as e:
            raise RuntimeError(f"메타데이터 로딩 실패: {e}")
    
    def _compute_statistics(self):
        """데이터셋 통계 정보 계산"""
        self.num_samples = len(self.metadata)
        self.unique_words = self.metadata['word_label'].nunique()
        self.word_counts = self.metadata['word_label'].value_counts()
        
        # 시퀀스 길이 통계
        sequence_lengths = self.metadata['original_length'].values
        self.avg_sequence_length = np.mean(sequence_lengths)
        self.min_sequence_length = np.min(sequence_lengths)
        self.max_sequence_length_actual = np.max(sequence_lengths)
        
        self.logger.info(f"📊 {self.split} 데이터셋 통계:")
        self.logger.info(f"  - 총 샘플 수: {self.num_samples}")
        self.logger.info(f"  - 고유 단어 수: {self.unique_words}")
        self.logger.info(f"  - 평균 시퀀스 길이: {self.avg_sequence_length:.1f}")
        self.logger.info(f"  - 시퀀스 길이 범위: {self.min_sequence_length}~{self.max_sequence_length_actual}")
    
    def _load_sequence(self, word_id: int) -> np.ndarray:
        """시퀀스 파일 로딩"""
        sequence_file = self.sequences_path / f"WORD{word_id:04d}_sequence.npy"
        
        if not sequence_file.exists():
            raise FileNotFoundError(f"시퀀스 파일이 없습니다: {sequence_file}")
        
        try:
            sequence = np.load(sequence_file)
            return sequence.astype(np.float32)
            
        except Exception as e:
            raise RuntimeError(f"시퀀스 로딩 실패 ({word_id}): {e}")
    
    def _normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """시퀀스 정규화"""
        if not self.normalize:
            return sequence
        
        # 키포인트별로 정규화 (x, y, confidence 구조 고려)
        normalized = sequence.copy()
        
        # 각 프레임별로 처리
        for frame_idx in range(len(sequence)):
            frame = sequence[frame_idx]
            
            # 3개씩 묶어서 처리 (x, y, confidence)
            for i in range(0, len(frame), 3):
                if i + 2 < len(frame):
                    # x, y 좌표는 0-1 범위로 클리핑
                    normalized[frame_idx][i] = np.clip(frame[i], 0, 1)      # x
                    normalized[frame_idx][i+1] = np.clip(frame[i+1], 0, 1)  # y
                    # confidence는 0-1 범위로 클리핑
                    normalized[frame_idx][i+2] = np.clip(frame[i+2], 0, 1)  # conf
        
        return normalized
    
    def _augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """데이터 증강 (학습시에만 적용)"""
        if not self.augment:
            return sequence
        
        augmented = sequence.copy()
        
        # 1. 시간축 노이즈 (프레임 순서 약간 변경)
        if random.random() < 0.3 and len(sequence) > 5:
            # 처음과 끝 몇 프레임은 건드리지 않고 중간만 조금 섞기
            start_idx = 2
            end_idx = len(sequence) - 2
            if end_idx > start_idx:
                # 중간 구간에서 1-2프레임 정도만 위치 바꾸기
                swap_idx1 = random.randint(start_idx, end_idx-1)
                swap_idx2 = min(swap_idx1 + 1, end_idx)
                augmented[swap_idx1], augmented[swap_idx2] = \
                    augmented[swap_idx2].copy(), augmented[swap_idx1].copy()
        
        # 2. 키포인트 노이즈 (작은 랜덤 노이즈 추가)
        if random.random() < 0.5:
            noise_scale = 0.02  # 2% 노이즈
            noise = np.random.normal(0, noise_scale, augmented.shape)
            
            # confidence 값에는 노이즈 추가하지 않음
            for i in range(2, augmented.shape[1], 3):  # confidence 인덱스들
                noise[:, i] = 0
            
            augmented = augmented + noise
            
            # 범위 클리핑
            augmented = np.clip(augmented, 0, 1)
        
        # 3. 시간축 스케일링 (속도 변화 시뮬레이션)
        if random.random() < 0.3:
            scale_factor = random.uniform(0.9, 1.1)  # ±10% 속도 변화
            
            original_length = len(sequence)
            new_length = int(original_length * scale_factor)
            new_length = max(5, min(new_length, self.max_sequence_length))
            
            if new_length != original_length:
                # 선형 보간으로 리샘플링
                indices = np.linspace(0, original_length-1, new_length)
                resampled = np.zeros((new_length, augmented.shape[1]))
                
                for i, idx in enumerate(indices):
                    lower_idx = int(np.floor(idx))
                    upper_idx = min(int(np.ceil(idx)), original_length-1)
                    
                    if lower_idx == upper_idx:
                        resampled[i] = augmented[lower_idx]
                    else:
                        # 선형 보간
                        weight = idx - lower_idx
                        resampled[i] = (1 - weight) * augmented[lower_idx] + \
                                      weight * augmented[upper_idx]
                
                augmented = resampled
        
        return augmented
    
    def _pad_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """시퀀스 패딩"""
        return pad_sequence(sequence, self.max_sequence_length, pad_value=0.0)
    
    def __len__(self) -> int:
        """데이터셋 크기 반환"""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """단일 샘플 반환"""
        if idx >= len(self.metadata):
            raise IndexError(f"인덱스 범위 초과: {idx} >= {len(self.metadata)}")
        
        # 메타데이터 추출
        sample_info = self.metadata.iloc[idx]
        word_id = sample_info['word_id']
        word_label = sample_info['word_label']
        original_length = sample_info['original_length']
        
        # 시퀀스 로딩
        try:
            sequence = self._load_sequence(word_id)
        except Exception as e:
            self.logger.error(f"시퀀스 로딩 실패 (idx={idx}, word_id={word_id}): {e}")
            # 빈 시퀀스로 대체
            sequence = np.zeros((self.max_sequence_length, 1629), dtype=np.float32)
            original_length = 0
        
        # 전처리 파이프라인
        sequence = self._normalize_sequence(sequence)
        sequence = self._augment_sequence(sequence)
        sequence = self._pad_sequence(sequence)
        
        # 라벨 인덱스 변환
        label_idx = self.vocab.word_to_index(word_label)
        
        # 텐서 변환
        sequence_tensor = torch.from_numpy(sequence).float()
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        
        # 추가 정보
        sample = {
            'sequence': sequence_tensor,        # [max_seq_len, keypoint_dim]
            'label': label_tensor,              # [1] - 라벨 인덱스
            'word_id': torch.tensor(word_id, dtype=torch.long),
            'word_label': word_label,           # 원본 단어 (문자열)
            'original_length': torch.tensor(original_length, dtype=torch.long),
            'sequence_mask': torch.tensor(
                [1.0 if i < original_length else 0.0 for i in range(self.max_sequence_length)], 
                dtype=torch.float
            )  # 패딩 마스크
        }
        
        return sample
    
    def get_word_distribution(self) -> Dict[str, int]:
        """단어별 분포 반환"""
        return self.word_counts.to_dict()
    
    def get_sample_by_word(self, word_label: str, limit: int = 5) -> List[Dict]:
        """특정 단어의 샘플들 반환"""
        word_samples = self.metadata[self.metadata['word_label'] == word_label]
        
        samples = []
        for idx in word_samples.index[:limit]:
            dataset_idx = self.metadata.index.get_loc(idx)
            sample = self[dataset_idx]
            samples.append(sample)
        
        return samples
    
    def get_class_weights(self) -> torch.Tensor:
        """클래스 불균형 해결을 위한 가중치 계산"""
        class_counts = np.zeros(len(self.vocab))
        
        for word_label in self.metadata['word_label']:
            label_idx = self.vocab.word_to_index(word_label)
            class_counts[label_idx] += 1
        
        # 역빈도 가중치 계산
        total_samples = len(self.metadata)
        weights = total_samples / (len(self.vocab) * class_counts + 1e-8)  # zero division 방지
        
        return torch.from_numpy(weights).float()
    
    def print_dataset_info(self):
        """데이터셋 정보 출력"""
        print("=" * 60)
        print(f"📊 {self.split.upper()} 데이터셋 정보")
        print("=" * 60)
        print(f"🔢 총 샘플 수: {self.num_samples:,}")
        print(f"📚 고유 단어 수: {self.unique_words}")
        print(f"📏 시퀀스 길이: 평균 {self.avg_sequence_length:.1f}, 범위 {self.min_sequence_length}~{self.max_sequence_length_actual}")
        print(f"🎯 최대 패딩 길이: {self.max_sequence_length}")
        print(f"🔄 데이터 증강: {'ON' if self.augment else 'OFF'}")
        print(f"📐 정규화: {'ON' if self.normalize else 'OFF'}")
        print()
        
        # 상위 10개 단어 분포
        print("🏆 상위 10개 단어 분포:")
        top_words = self.word_counts.head(10)
        for word, count in top_words.items():
            percentage = (count / self.num_samples) * 100
            print(f"  {word}: {count}개 ({percentage:.1f}%)")
        
        print("=" * 60)

def create_data_loaders(train_batch_size: int = BATCH_SIZE,
                       val_batch_size: int = BATCH_SIZE,
                       num_workers: int = 4,
                       pin_memory: bool = True,
                       vocab_path: Optional[Path] = None,
                       max_sequence_length: int = 200) -> Tuple[DataLoader, DataLoader]:
    """
    학습/검증용 DataLoader 생성
    
    Args:
        train_batch_size: 학습용 배치 크기
        val_batch_size: 검증용 배치 크기  
        num_workers: 멀티프로세싱 워커 수
        pin_memory: GPU 메모리 고정 사용 여부
        vocab_path: 사전 파일 경로
        max_sequence_length: 최대 시퀀스 길이
    
    Returns:
        (train_loader, val_loader)
    """
    logger = setup_logger("DataLoader")
    
    # 데이터셋 생성
    train_dataset = SignLanguageDataset(
        split="train",
        vocab_path=vocab_path,
        max_sequence_length=max_sequence_length,
        augment=True,
        normalize=True
    )
    
    val_dataset = SignLanguageDataset(
        split="val", 
        vocab_path=vocab_path,
        max_sequence_length=max_sequence_length,
        augment=False,
        normalize=True
    )
    
    logger.info(f"데이터셋 생성 완료: 학습 {len(train_dataset)}, 검증 {len(val_dataset)}")
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # 배치 크기 일관성을 위해
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0
    )
    
    logger.info(f"DataLoader 생성 완료: 학습 {len(train_loader)} 배치, 검증 {len(val_loader)} 배치")
    
    return train_loader, val_loader

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    배치 데이터 결합 함수 (필요시 사용)
    
    Args:
        batch: 샘플들의 리스트
        
    Returns:
        배치 텐서들의 딕셔너리
    """
    # 기본적으로는 PyTorch의 default_collate가 잘 작동하지만,
    # 복잡한 배치 처리가 필요한 경우 여기서 구현
    
    sequences = torch.stack([item['sequence'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    word_ids = torch.stack([item['word_id'] for item in batch])
    original_lengths = torch.stack([item['original_length'] for item in batch])
    sequence_masks = torch.stack([item['sequence_mask'] for item in batch])
    
    # 문자열은 리스트로 유지
    word_labels = [item['word_label'] for item in batch]
    
    return {
        'sequence': sequences,
        'label': labels,
        'word_id': word_ids,
        'word_label': word_labels,
        'original_length': original_lengths,
        'sequence_mask': sequence_masks
    }

def main():
    """테스트용 메인 함수"""
    print("🧪 데이터셋 테스트")
    
    try:
        # 데이터셋 생성
        train_dataset = SignLanguageDataset(split="train", augment=True)
        val_dataset = SignLanguageDataset(split="val", augment=False)
        
        # 정보 출력
        train_dataset.print_dataset_info()
        val_dataset.print_dataset_info()
        
        # 샘플 테스트
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"\n📝 샘플 테스트:")
            print(f"  시퀀스 크기: {sample['sequence'].shape}")
            print(f"  라벨: {sample['label'].item()} -> '{sample['word_label']}'")
            print(f"  원본 길이: {sample['original_length'].item()}")
            print(f"  단어 ID: {sample['word_id'].item()}")
        
        # DataLoader 테스트
        train_loader, val_loader = create_data_loaders(
            train_batch_size=8, val_batch_size=8, num_workers=0
        )
        
        print(f"\n🔄 DataLoader 테스트:")
        batch = next(iter(train_loader))
        print(f"  배치 시퀀스 크기: {batch['sequence'].shape}")
        print(f"  배치 라벨 크기: {batch['label'].shape}")
        print(f"  첫 번째 샘플 라벨: '{batch['word_label'][0]}'")
        
        print("\n✅ 데이터셋 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 데이터셋 테스트 실패: {e}")

if __name__ == "__main__":
    main()