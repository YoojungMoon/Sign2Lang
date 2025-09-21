#!/usr/bin/env python3
"""
SU:DA - 실시간 수어 인식 추론
웹캠을 통한 실시간 수어 동작 인식 및 텍스트 변환
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
import time
import threading
import queue
from datetime import datetime

from utils import (
    setup_logger, load_checkpoint, get_device, normalize_keypoints,
    pad_sequence, CHECKPOINTS_PATH, TOTAL_LANDMARKS
)
from model import create_model
from vocab import SignVocabulary

class MediaPipeExtractor:
    """MediaPipe를 이용한 실시간 키포인트 추출"""
    
    def __init__(self):
        self.logger = setup_logger("MediaPipeExtractor")
        
        # MediaPipe 초기화
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Holistic 모델 초기화
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.logger.info("MediaPipe Holistic 모델 초기화 완료")
    
    def extract_keypoints(self, image: np.ndarray) -> np.ndarray:
        """이미지에서 키포인트 추출"""
        # BGR을 RGB로 변환
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        
        # MediaPipe 처리
        results = self.holistic.process(rgb_image)
        
        # 키포인트 추출
        keypoints = self._extract_landmarks(results)
        
        return keypoints, results
    
    def _extract_landmarks(self, results) -> np.ndarray:
        """랜드마크 결과에서 키포인트 배열 추출"""
        keypoints = []
        
        # Pose landmarks (33개)
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.visibility])
        else:
            keypoints.extend([0.0] * (33 * 3))
        
        # Left hand landmarks (21개)
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.visibility])
        else:
            keypoints.extend([0.0] * (21 * 3))
        
        # Right hand landmarks (21개)
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.visibility])
        else:
            keypoints.extend([0.0] * (21 * 3))
        
        # Face landmarks (468개)
        if results.face_landmarks:
            for landmark in results.face_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.visibility])
        else:
            keypoints.extend([0.0] * (468 * 3))
        
        return np.array(keypoints, dtype=np.float32)
    
    def draw_landmarks(self, image: np.ndarray, results) -> np.ndarray:
        """이미지에 랜드마크 그리기"""
        image.flags.writeable = True
        
        # Pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Hands
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Face (선택적 - 너무 많아서 생략 가능)
        # if results.face_landmarks:
        #     self.mp_drawing.draw_landmarks(
        #         image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
        #         None, self.mp_drawing_styles.get_default_face_mesh_contours_style()
        #     )
        
        return image
    
    def __del__(self):
        """소멸자"""
        if hasattr(self, 'holistic'):
            self.holistic.close()

class SignLanguageInference:
    """실시간 수어 인식 추론 시스템"""
    
    def __init__(self,
                 checkpoint_path: Optional[Path] = None,
                 vocab_path: Optional[Path] = None,
                 sequence_length: int = 60,
                 confidence_threshold: float = 0.7,
                 detection_cooldown: float = 2.0,
                 device: Optional[torch.device] = None):
        """
        Args:
            checkpoint_path: 모델 체크포인트 경로
            vocab_path: 사전 파일 경로  
            sequence_length: 분석할 시퀀스 길이 (프레임 수)
            confidence_threshold: 인식 임계값
            detection_cooldown: 같은 단어 연속 인식 방지 시간(초)
            device: 추론 디바이스
        """
        self.logger = setup_logger("SignLanguageInference")
        
        # 설정
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        self.detection_cooldown = detection_cooldown
        self.device = device if device is not None else get_device()
        
        # MediaPipe 초기화
        self.mp_extractor = MediaPipeExtractor()
        
        # 사전 로딩
        self.vocab = SignVocabulary(vocab_path)
        self.vocab_size = len(self.vocab)
        
        # 모델 로딩
        self.model = create_model(vocab_size=self.vocab_size, device=self.device)
        self._load_model(checkpoint_path)
        
        # 시퀀스 버퍼 (최근 프레임들 저장)
        self.keypoint_buffer: Deque[np.ndarray] = deque(maxlen=sequence_length)
        
        # 인식 결과 관리
        self.last_detection_time = 0
        self.last_detected_word = ""
        self.detection_history = deque(maxlen=10)  # 최근 10개 결과
        
        # 통계
        self.frame_count = 0
        self.detection_count = 0
        self.fps_counter = deque(maxlen=30)
        
        self.logger.info("=" * 60)
        self.logger.info("🎯 실시간 수어 인식 시스템 초기화 완료")
        self.logger.info("=" * 60)
        self.logger.info(f"📚 어휘 크기: {self.vocab_size}")
        self.logger.info(f"🎬 시퀀스 길이: {sequence_length} 프레임")
        self.logger.info(f"🎯 신뢰도 임계값: {confidence_threshold}")
        self.logger.info(f"⏱️ 인식 쿨다운: {detection_cooldown}초")
        self.logger.info(f"💾 디바이스: {self.device}")
        self.logger.info("=" * 60)
    
    def _load_model(self, checkpoint_path: Optional[Path]):
        """모델 로딩"""
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINTS_PATH / "best.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트 파일이 없습니다: {checkpoint_path}")
        
        try:
            checkpoint = load_checkpoint(checkpoint_path, self.model.model)
            self.model.eval()
            
            self.logger.info(f"✅ 모델 로딩 완료")
            self.logger.info(f"  - 체크포인트: {checkpoint_path}")
            self.logger.info(f"  - 검증 정확도: {checkpoint['accuracy']:.2f}%")
            
        except Exception as e:
            self.logger.error(f"모델 로딩 실패: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """단일 프레임 처리"""
        self.frame_count += 1
        frame_start_time = time.time()
        
        # 키포인트 추출
        keypoints, mp_results = self.mp_extractor.extract_keypoints(frame)
        
        # 정규화
        normalized_keypoints = normalize_keypoints(keypoints)
        
        # 버퍼에 추가
        self.keypoint_buffer.append(normalized_keypoints)
        
        # 시각화
        annotated_frame = self.mp_extractor.draw_landmarks(frame, mp_results)
        
        # 수어 인식 시도
        recognition_result = self._try_recognition()
        
        # UI 정보 추가
        annotated_frame = self._add_ui_overlay(annotated_frame, recognition_result)
        
        # FPS 계산
        frame_time = time.time() - frame_start_time
        self.fps_counter.append(1.0 / frame_time if frame_time > 0 else 0)
        
        result_info = {
            'frame_count': self.frame_count,
            'buffer_size': len(self.keypoint_buffer),
            'fps': np.mean(self.fps_counter) if self.fps_counter else 0,
            'keypoints_detected': np.any(keypoints != 0),
            **recognition_result
        }
        
        return annotated_frame, result_info
    
    def _try_recognition(self) -> Dict:
        """수어 인식 시도"""
        current_time = time.time()
        
        # 버퍼가 충분히 찼는지 확인
        if len(self.keypoint_buffer) < self.sequence_length:
            return {
                'word_detected': False,
                'word': '',
                'confidence': 0.0,
                'top3_predictions': [],
                'status': 'collecting_frames'
            }
        
        # 키포인트가 감지되었는지 확인
        recent_keypoints = list(self.keypoint_buffer)[-30:]  # 최근 30프레임
        if not any(np.any(kp != 0) for kp in recent_keypoints):
            return {
                'word_detected': False,
                'word': '',
                'confidence': 0.0,
                'top3_predictions': [],
                'status': 'no_person_detected'
            }
        
        try:
            # 시퀀스 준비
            sequence = np.array(list(self.keypoint_buffer))
            sequence = pad_sequence(sequence, self.sequence_length)
            
            # 텐서 변환
            sequence_tensor = torch.from_numpy(sequence).unsqueeze(0).float().to(self.device)
            
            # 추론
            self.model.eval()
            with torch.no_grad():
                batch = {
                    'sequence': sequence_tensor,
                    'sequence_mask': torch.ones(1, self.sequence_length).to(self.device)
                }
                pred_outputs = self.model.predict_batch(batch)
            
            # 결과 처리
            probabilities = pred_outputs['probabilities'][0].cpu().numpy()
            predicted_idx = pred_outputs['predictions'][0].item()
            max_confidence = pred_outputs['max_probabilities'][0].item()
            
            # Top-3 예측
            top3_indices = np.argsort(probabilities)[-3:][::-1]
            top3_predictions = [
                {
                    'word': self.vocab.index_to_word(idx),
                    'confidence': probabilities[idx] * 100,
                    'index': int(idx)
                }
                for idx in top3_indices
            ]
            
            predicted_word = self.vocab.index_to_word(predicted_idx)
            
            # 인식 결정
            is_confident = max_confidence >= self.confidence_threshold
            is_not_cooldown = (current_time - self.last_detection_time) >= self.detection_cooldown
            is_different_word = predicted_word != self.last_detected_word
            
            word_detected = is_confident and (is_not_cooldown or is_different_word)
            
            if word_detected:
                self.last_detection_time = current_time
                self.last_detected_word = predicted_word
                self.detection_count += 1
                
                # 인식 기록 저장
                self.detection_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'word': predicted_word,
                    'confidence': max_confidence * 100,
                    'frame_count': self.frame_count
                })
                
                self.logger.info(f"🎯 수어 인식: '{predicted_word}' (신뢰도: {max_confidence*100:.1f}%)")
            
            return {
                'word_detected': word_detected,
                'word': predicted_word if word_detected else '',
                'confidence': max_confidence * 100,
                'top3_predictions': top3_predictions,
                'status': 'recognized' if word_detected else 'analyzing',
                'raw_confidence': max_confidence * 100,
                'raw_prediction': predicted_word
            }
            
        except Exception as e:
            self.logger.error(f"인식 중 오류: {e}")
            return {
                'word_detected': False,
                'word': '',
                'confidence': 0.0,
                'top3_predictions': [],
                'status': 'error',
                'error': str(e)
            }
    
    def _add_ui_overlay(self, frame: np.ndarray, recognition_result: Dict) -> np.ndarray:
        """UI 오버레이 추가"""
        height, width = frame.shape[:2]
        
        # 반투명 배경
        overlay = frame.copy()
        
        # 상단 정보 패널
        cv2.rectangle(overlay, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 제목
        cv2.putText(frame, "SU:DA - Real-time Sign Language Recognition", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 상태 정보
        status_text = f"Buffer: {len(self.keypoint_buffer)}/{self.sequence_length} | "
        status_text += f"FPS: {np.mean(self.fps_counter):.1f} | " if self.fps_counter else "FPS: 0 | "
        status_text += f"Detections: {self.detection_count}"
        
        cv2.putText(frame, status_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 인식 결과
        if recognition_result.get('word_detected', False):
            # 인식된 단어 (크게)
            word = recognition_result['word']
            confidence = recognition_result['confidence']
            
            # 성공 배경
            cv2.rectangle(frame, (10, 70), (width-10, 110), (0, 255, 0), 2)
            cv2.putText(frame, f"Detected: {word}", (20, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.1f}%", (20, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        elif recognition_result.get('status') == 'analyzing':
            # 분석 중
            raw_word = recognition_result.get('raw_prediction', '')
            raw_conf = recognition_result.get('raw_confidence', 0)
            
            cv2.rectangle(frame, (10, 70), (width-10, 110), (0, 255, 255), 1)
            cv2.putText(frame, f"Analyzing: {raw_word}", (20, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {raw_conf:.1f}%", (20, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        else:
            # 대기 중
            status_msg = {
                'collecting_frames': 'Collecting frames...',
                'no_person_detected': 'No person detected',
                'error': 'Processing error'
            }.get(recognition_result.get('status', ''), 'Ready')
            
            cv2.putText(frame, f"Status: {status_msg}", (20, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 우측 Top-3 예측 (작게)
        if recognition_result.get('top3_predictions'):
            y_offset = 140
            cv2.putText(frame, "Top-3 Predictions:", (width-250, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            for i, pred in enumerate(recognition_result['top3_predictions'][:3]):
                y_pos = y_offset + 25 + (i * 20)
                text = f"{i+1}. {pred['word']} ({pred['confidence']:.1f}%)"
                color = (0, 255, 0) if i == 0 else (200, 200, 200)
                cv2.putText(frame, text, (width-240, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 최근 인식 기록 (하단)
        if self.detection_history:
            cv2.putText(frame, "Recent detections:", (10, height-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            recent = list(self.detection_history)[-3:]  # 최근 3개
            for i, detection in enumerate(recent):
                y_pos = height - 40 + (i * 15)
                time_str = detection['timestamp'][-8:-3]  # HH:MM만
                text = f"{time_str} - {detection['word']} ({detection['confidence']:.1f}%)"
                cv2.putText(frame, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # 조작 안내
        cv2.putText(frame, "Press 'q' to quit, 'r' to reset, 's' to save", 
                   (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def reset_buffer(self):
        """버퍼 초기화"""
        self.keypoint_buffer.clear()
        self.last_detection_time = 0
        self.last_detected_word = ""
        self.logger.info("🔄 버퍼 초기화")
    
    def save_detection_log(self):
        """인식 기록 저장"""
        if not self.detection_history:
            self.logger.info("저장할 인식 기록이 없습니다.")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(f"sign_detection_log_{timestamp}.txt")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"SU:DA 수어 인식 로그 - {datetime.now()}\n")
            f.write("=" * 50 + "\n")
            f.write(f"총 프레임 수: {self.frame_count}\n")
            f.write(f"총 인식 수: {self.detection_count}\n")
            f.write(f"인식 성공률: {self.detection_count/self.frame_count*100:.2f}%\n\n")
            
            f.write("인식 기록:\n")
            for detection in self.detection_history:
                f.write(f"{detection['timestamp']} - {detection['word']} "
                       f"({detection['confidence']:.1f}%) [Frame: {detection['frame_count']}]\n")
        
        self.logger.info(f"📁 인식 로그 저장: {log_file}")
    
    def run_webcam_inference(self, camera_id: int = 0, window_size: Tuple[int, int] = (1280, 720)):
        """웹캠 실시간 추론 실행"""
        self.logger.info(f"🎥 웹캠 추론 시작 (카메라 ID: {camera_id})")
        
        # 웹캠 초기화
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_size[1])
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            raise RuntimeError(f"웹캠을 열 수 없습니다 (ID: {camera_id})")
        
        self.logger.info("웹캠 연결 성공! 실시간 수어 인식을 시작합니다.")
        self.logger.info("조작법: 'q'=종료, 'r'=버퍼초기화, 's'=로그저장")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("프레임을 읽을 수 없습니다.")
                    break
                
                # 좌우 반전 (거울 효과)
                frame = cv2.flip(frame, 1)
                
                # 프레임 처리
                processed_frame, result_info = self.process_frame(frame)
                
                # 화면 출력
                cv2.imshow('SU:DA - Sign Language Recognition', processed_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    self.logger.info("사용자 종료 요청")
                    break
                elif key == ord('r'):
                    self.reset_buffer()
                elif key == ord('s'):
                    self.save_detection_log()
                elif key == ord(' '):  # 스페이스바로 일시정지
                    cv2.waitKey(0)
        
        except KeyboardInterrupt:
            self.logger.info("키보드 인터럽트로 종료")
        
        except Exception as e:
            self.logger.error(f"추론 중 오류: {e}")
        
        finally:
            # 정리
            cap.release()
            cv2.destroyAllWindows()
            
            # 최종 통계
            self.logger.info("=" * 60)
            self.logger.info("🎉 실시간 수어 인식 종료")
            self.logger.info("=" * 60)
            self.logger.info(f"📊 총 처리 프레임: {self.frame_count:,}")
            self.logger.info(f"🎯 총 인식 횟수: {self.detection_count}")
            if self.frame_count > 0:
                self.logger.info(f"📈 인식 비율: {self.detection_count/self.frame_count*100:.2f}%")
            if self.fps_counter:
                self.logger.info(f"⚡ 평균 FPS: {np.mean(self.fps_counter):.1f}")
            self.logger.info("=" * 60)
            
            # 자동 로그 저장
            if self.detection_history:
                self.save_detection_log()

def main():
    """메인 실행 함수"""
    try:
        # 추론 시스템 초기화
        inference_system = SignLanguageInference(
            checkpoint_path=CHECKPOINTS_PATH / "best.pt",
            sequence_length=60,  # 2초 @ 30fps
            confidence_threshold=0.7,
            detection_cooldown=1.5
        )
        
        # 웹캠 추론 실행
        inference_system.run_webcam_inference(
            camera_id=0,
            window_size=(1280, 720)
        )
        
        return True
        
    except Exception as e:
        print(f"❌ 추론 실행 실패: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ 실시간 수어 인식이 성공적으로 완료되었습니다!")
    else:
        print("❌ 실시간 수어 인식이 실패했습니다.")