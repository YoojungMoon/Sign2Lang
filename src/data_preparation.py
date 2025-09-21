#!/usr/bin/env python3
"""
SU:DA - 데이터 전처리 모듈
raw 데이터를 processed 형태로 변환
- 프레임별 키포인트를 시퀀스로 합치기
- 시간 정보를 프레임 인덱스로 변환
- train/val 분할
- vocab 생성
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Stratified split 시도 → 불가 시 랜덤 분할로 폴백
from sklearn.model_selection import train_test_split
import random

from utils import (
    setup_logger, load_json, save_json, get_video_fps, time_to_frame_index,
    extract_keypoints_from_json, normalize_keypoints, pad_sequence,
    DATA_RAW_PATH, DATA_PROCESSED_PATH, WORD_START, WORD_END, SPEAKER_ID, DIRECTION,
    get_word_id_from_filename, is_valid_word_id, create_word_range
)


class DataPreprocessor:
    """데이터 전처리 클래스"""

    def __init__(self):
        self.logger = setup_logger("DataPreprocessor", "data_preparation.log")
        self.word_range = create_word_range()
        self.max_sequence_length = 200  # 최대 시퀀스 길이 (프레임 수)

        # 경로 설정
        self.raw_keypoints_path = DATA_RAW_PATH / "keypoints"
        self.raw_labels_path = DATA_RAW_PATH / "labels"
        self.raw_videos_path = DATA_RAW_PATH / "videos"

        self.processed_sequences_path = DATA_PROCESSED_PATH / "sequences"
        self.processed_splits_path = DATA_PROCESSED_PATH / "splits"
        self.processed_vocab_path = DATA_PROCESSED_PATH / "vocab"

        self.logger.info("데이터 전처리 초기화 완료")
        self.logger.info(f"처리 대상: WORD{WORD_START}~{WORD_END} ({len(self.word_range)}개)")

    # ---------------------------
    # 로딩/전처리 유틸
    # ---------------------------
    def load_morpheme_data(self, word_id: int) -> Optional[Dict]:
        """형태소 데이터 로딩"""
        filename = f"NIA_SL_WORD{word_id:04d}_{SPEAKER_ID}_{DIRECTION}_morpheme.json"
        file_path = self.raw_labels_path / filename

        if not file_path.exists():
            self.logger.warning(f"형태소 파일 없음: {filename}")
            return None

        try:
            return load_json(file_path)
        except Exception as e:
            self.logger.error(f"형태소 데이터 로딩 실패 ({filename}): {e}")
            return None

    def load_keypoint_sequence(self, word_id: int) -> Optional[np.ndarray]:
        """프레임별 키포인트를 시퀀스로 합치기"""
        folder_name = f"NIA_SL_WORD{word_id:04d}_{SPEAKER_ID}_{DIRECTION}"
        keypoints_folder = self.raw_keypoints_path / folder_name

        if not keypoints_folder.exists():
            self.logger.warning(f"키포인트 폴더 없음: {folder_name}")
            return None

        # 프레임별 파일들 수집
        frame_files = []
        for file_path in keypoints_folder.glob("*.json"):
            if "_keypoints.json" in file_path.name:
                # 파일명에서 프레임 번호 추출 (…_000000000000_keypoints.json 형태)
                try:
                    frame_num_str = file_path.stem.split('_')[-2]
                    frame_num = int(frame_num_str)
                    frame_files.append((frame_num, file_path))
                except (ValueError, IndexError):
                    continue

        if not frame_files:
            self.logger.warning(f"키포인트 파일 없음: {folder_name}")
            return None

        # 프레임 번호 순으로 정렬
        frame_files.sort(key=lambda x: x[0])

        # 각 프레임의 키포인트 추출
        sequence_keypoints = []
        for frame_num, file_path in frame_files:
            try:
                keypoint_json = load_json(file_path)
                keypoints = extract_keypoints_from_json(keypoint_json)
                normalized_keypoints = normalize_keypoints(keypoints)
                sequence_keypoints.append(normalized_keypoints)
            except Exception as e:
                self.logger.warning(f"프레임 {frame_num} 키포인트 추출 실패: {e}")
                # 빈 키포인트로 대체 (543 landmarks * 3)
                sequence_keypoints.append(np.zeros(543 * 3, dtype=np.float32))

        if not sequence_keypoints:
            return None

        sequence = np.array(sequence_keypoints, dtype=np.float32)
        self.logger.debug(f"키포인트 시퀀스 생성: {folder_name}, shape={sequence.shape}")
        return sequence

    def get_video_metadata(self, word_id: int) -> Optional[Dict]:
        """비디오 메타데이터 추출 (현재는 FPS만 사용)"""
        filename = f"NIA_SL_WORD{word_id:04d}_{SPEAKER_ID}_{DIRECTION}.mp4"
        video_path = self.raw_videos_path / filename

        if not video_path.exists():
            self.logger.warning(f"비디오 파일 없음: {filename}")
            return None

        try:
            fps = get_video_fps(video_path)
            return {"fps": fps, "path": str(video_path)}
        except Exception as e:
            self.logger.error(f"비디오 메타데이터 추출 실패 ({filename}): {e}")
            return None

    # ---------------------------
    # 단일 샘플 처리
    # ---------------------------
    def process_single_word(self, word_id: int) -> Optional[Dict]:
        """단일 단어 데이터 처리"""
        # 1) 형태소
        morpheme_data = self.load_morpheme_data(word_id)
        if morpheme_data is None:
            return None

        # 2) 키포인트 시퀀스
        keypoint_sequence = self.load_keypoint_sequence(word_id)
        if keypoint_sequence is None:
            return None

        # 3) 비디오 메타데이터
        video_metadata = self.get_video_metadata(word_id)
        if video_metadata is None:
            return None

        # 4) 라벨/시간 정보
        try:
            data_entries = morpheme_data.get("data", [])
            if not data_entries:
                self.logger.warning(f"형태소 데이터 없음: WORD{word_id}")
                return None

            first_entry = data_entries[0]
            start_time = float(first_entry.get("start", 0))
            end_time = float(first_entry.get("end", 0))

            attributes = first_entry.get("attributes", [])
            if not attributes:
                self.logger.warning(f"수어 단어 라벨 없음: WORD{word_id}")
                return None

            word_label = attributes[0].get("name", "")
            if not word_label:
                self.logger.warning(f"수어 단어명 없음: WORD{word_id}")
                return None
        except (KeyError, ValueError, IndexError) as e:
            self.logger.error(f"형태소 데이터 파싱 실패 WORD{word_id}: {e}")
            return None

        # 5) 시간을 프레임 인덱스로 변환
        fps = video_metadata["fps"]
        start_frame = time_to_frame_index(start_time, fps)
        end_frame = time_to_frame_index(end_time, fps)

        # 프레임 범위 검증
        seq_len = len(keypoint_sequence)
        start_frame = max(0, min(start_frame, seq_len - 1))
        end_frame = max(start_frame + 1, min(end_frame, seq_len))

        # 6) 시퀀스 패딩 (모델 입력 길이 통일)
        padded_sequence = pad_sequence(keypoint_sequence, self.max_sequence_length)

        # 7) 결과 반환 (시퀀스는 별도 파일로 저장하므로 메타와 함께 반환)
        return {
            "word_id": word_id,
            "word_label": word_label,
            "sequence": padded_sequence,
            "original_length": seq_len,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_time,
            "end_time": end_time,
            "fps": fps,
            "video_duration": morpheme_data.get("metaData", {}).get("duration", 0),
        }

    def save_processed_sequence(self, word_id: int, sequence: np.ndarray):
        """처리된 시퀀스를 .npy 파일로 저장"""
        filename = f"WORD{word_id:04d}_sequence.npy"
        filepath = self.processed_sequences_path / filename
        self.processed_sequences_path.mkdir(parents=True, exist_ok=True)
        np.save(filepath, sequence)

    # ---------------------------
    # 다수 샘플 처리/어휘집/분할
    # ---------------------------
    def process_all_words(self) -> List[Dict]:
        """모든 단어 데이터 처리"""
        self.logger.info("전체 단어 데이터 처리 시작")

        processed_samples: List[Dict] = []
        failed_words: List[int] = []

        for word_id in tqdm(self.word_range, desc="데이터 처리 중"):
            if not is_valid_word_id(word_id):
                continue

            processed_data = self.process_single_word(word_id)
            if processed_data is not None:
                # 시퀀스는 파일로 저장하고, 메타데이터만 리스트에 유지
                self.save_processed_sequence(word_id, processed_data["sequence"])
                metadata = {k: v for k, v in processed_data.items() if k != "sequence"}
                processed_samples.append(metadata)
            else:
                failed_words.append(word_id)

        self.logger.info(f"데이터 처리 완료: 성공 {len(processed_samples)}개, 실패 {len(failed_words)}개")
        if failed_words:
            self.logger.warning(f"처리 실패한 단어들: {failed_words[:10]}...")  # 처음 10개만 표시

        return processed_samples

    def create_vocabulary(self, processed_samples: List[Dict]) -> Dict[str, int]:
        """수어 단어 사전 생성 및 저장 (json/txt)"""
        self.logger.info("수어 단어 사전 생성 중")

        unique_words = {s["word_label"] for s in processed_samples}

        # 특수 토큰
        vocab: Dict[str, int] = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<SOS>": 2,
            "<EOS>": 3,
        }

        # 단어 사전은 정렬 후 4부터 부여
        for i, word in enumerate(sorted(unique_words)):
            vocab[word] = i + 4

        self.logger.info(f"수어 단어 사전 생성 완료: {len(vocab)}개 단어 (특수토큰 4개 포함)")

        self.processed_vocab_path.mkdir(parents=True, exist_ok=True)
        save_json(vocab, self.processed_vocab_path / "vocab.json")

        # 사람이 보기 좋은 txt 버전도 저장
        with open(self.processed_vocab_path / "vocab.txt", "w", encoding="utf-8") as f:
            for word, idx in sorted(vocab.items(), key=lambda x: x[1]):
                f.write(f"{idx}\t{word}\n")

        return vocab

    def split_train_val(
        self,
        processed_samples: List[Dict],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[List[Dict], List[Dict]]:
        """학습용/검증용 데이터 분할 (가능하면 stratify, 아니면 랜덤)"""
        self.logger.info(f"데이터 분할 중 (검증용 비율: {test_size})")

        labels = [s["word_label"] for s in processed_samples]

        # 우선 stratified split을 시도
        try:
            train_samples, val_samples = train_test_split(
                processed_samples,
                test_size=test_size,
                random_state=random_state,
                stratify=labels,
                shuffle=True,
            )
        except ValueError:
            # 클래스별 샘플 수가 1개인 경우 등 → 랜덤 분할로 폴백
            from collections import Counter

            label_counts = Counter(labels)
            single_sample_classes = [lab for lab, cnt in label_counts.items() if cnt == 1]
            if single_sample_classes:
                self.logger.warning(f"1개 샘플만 있는 클래스들: {len(single_sample_classes)}개")
                self.logger.warning(f"예시: {single_sample_classes[:5]}")

            random.seed(random_state)
            shuffled = processed_samples.copy()
            random.shuffle(shuffled)

            split_idx = int(len(shuffled) * (1 - test_size))
            train_samples = shuffled[:split_idx]
            val_samples = shuffled[split_idx:]

        self.logger.info(f"데이터 분할 완료: 학습용 {len(train_samples)}개, 검증용 {len(val_samples)}개")
        return train_samples, val_samples

    def save_splits(self, train_samples: List[Dict], val_samples: List[Dict]) -> Dict:
        """분할된 데이터 CSV + 요약 저장"""
        self.processed_splits_path.mkdir(parents=True, exist_ok=True)

        # CSV 저장
        train_df = pd.DataFrame(train_samples)
        val_df = pd.DataFrame(val_samples)
        train_file = self.processed_splits_path / "train.csv"
        val_file = self.processed_splits_path / "val.csv"
        train_df.to_csv(train_file, index=False, encoding="utf-8")
        val_df.to_csv(val_file, index=False, encoding="utf-8")
        self.logger.info(f"분할 데이터 저장 완료: {train_file}, {val_file}")

        # 요약 저장
        total = len(train_samples) + len(val_samples)
        summary = {
            "total_samples": total,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "train_ratio": len(train_samples) / max(total, 1),
            "val_ratio": len(val_samples) / max(total, 1),
            "unique_words": len(set(s["word_label"] for s in train_samples + val_samples)),
            "max_sequence_length": self.max_sequence_length,
            "word_range": f"{WORD_START}-{WORD_END}",
            "speaker": SPEAKER_ID,
            "direction": DIRECTION,
        }
        save_json(summary, self.processed_splits_path / "summary.json")
        return summary

    # ---------------------------
    # 파이프라인
    # ---------------------------
    def run_preprocessing(self) -> bool:
        """전체 전처리 파이프라인 실행"""
        self.logger.info("=" * 60)
        self.logger.info("🚀 SU:DA 데이터 전처리 시작")
        self.logger.info("=" * 60)

        try:
            # 1) 전체 단어 처리
            processed_samples = self.process_all_words()
            if not processed_samples:
                self.logger.error("처리된 데이터가 없습니다!")
                return False

            # 2) 단어 사전
            _ = self.create_vocabulary(processed_samples)

            # 3) 학습/검증 분할
            train_samples, val_samples = self.split_train_val(processed_samples)

            # 4) 분할 저장 + 요약
            summary = self.save_splits(train_samples, val_samples)

            # 5) 로그 요약
            self.logger.info("=" * 60)
            self.logger.info("✅ 데이터 전처리 완료!")
            self.logger.info("=" * 60)
            self.logger.info(f"📊 총 샘플 수: {summary['total_samples']}")
            self.logger.info(f"🎯 학습용: {summary['train_samples']}개 ({summary['train_ratio']:.1%})")
            self.logger.info(f"🔍 검증용: {summary['val_samples']}개 ({summary['val_ratio']:.1%})")
            self.logger.info(f"📚 수어 단어 수: {summary['unique_words']}개")
            self.logger.info(f"📏 최대 시퀀스 길이: {summary['max_sequence_length']}")
            self.logger.info(f"💾 저장 위치: {DATA_PROCESSED_PATH}")
            self.logger.info("=" * 60)
            return True

        except Exception as e:
            self.logger.error(f"데이터 전처리 실패: {e}")
            return False


def main():
    """단독 실행 시 편의용"""
    preprocessor = DataPreprocessor()
    ok = preprocessor.run_preprocessing()
    if ok:
        print("🎉 데이터 전처리가 성공적으로 완료되었습니다!")
    else:
        print("❌ 데이터 전처리가 실패했습니다.")


if __name__ == "__main__":
    main()
