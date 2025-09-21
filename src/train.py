#!/usr/bin/env python3
"""
SU:DA - 수어 인식 모델 학습
BiLSTM 모델 학습 및 검증
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from utils import (
    setup_logger, save_checkpoint, load_checkpoint, get_device,
    LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, CHECKPOINTS_PATH, LOGS_PATH,
    format_time, count_parameters
)
from model import create_model
from dataset import create_data_loaders
from vocab import SignVocabulary

class SignLanguageTrainer:
    """수어 인식 모델 학습 클래스"""
    
    def __init__(self,
                 vocab_path: Optional[Path] = None,
                 learning_rate: float = LEARNING_RATE,
                 batch_size: int = BATCH_SIZE,
                 num_epochs: int = NUM_EPOCHS,
                 device: Optional[torch.device] = None,
                 resume_from: Optional[Path] = None,
                 use_class_weights: bool = True,
                 gradient_clip_val: float = 1.0,
                 save_every: int = 5):
        """
        Args:
            vocab_path: 사전 파일 경로
            learning_rate: 학습률
            batch_size: 배치 크기
            num_epochs: 학습 에포크 수
            device: 학습 디바이스
            resume_from: 재개할 체크포인트 경로
            use_class_weights: 클래스 가중치 사용 여부
            gradient_clip_val: 기울기 클리핑 값
            save_every: 체크포인트 저장 주기
        """
        self.logger = setup_logger("SignLanguageTrainer", "train.log")
        
        # 하이퍼파라미터
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.gradient_clip_val = gradient_clip_val
        self.save_every = save_every
        self.use_class_weights = use_class_weights
        
        # 디바이스 설정
        self.device = device if device is not None else get_device()
        
        # 사전 로딩
        self.vocab = SignVocabulary(vocab_path)
        self.vocab_size = len(self.vocab)
        
        # 모델 생성
        self.model = create_model(
            vocab_size=self.vocab_size,
            device=self.device,
            use_attention=True,
            use_positional_encoding=False
        )
        
        # 데이터 로더 생성
        self.train_loader, self.val_loader = create_data_loaders(
            train_batch_size=batch_size,
            val_batch_size=batch_size,
            vocab_path=vocab_path
        )
        
        # 클래스 가중치 설정
        if self.use_class_weights:
            class_weights = self.train_loader.dataset.get_class_weights()
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # 옵티마이저 설정
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True,
            min_lr=1e-7
        )
        
        # 학습 상태 초기화
        self.start_epoch = 0
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # TensorBoard 설정
        self.writer = SummaryWriter(log_dir=LOGS_PATH / "tensorboard")
        
        # 체크포인트 재개
        if resume_from is not None:
            self._load_checkpoint(resume_from)
        
        self.logger.info("=" * 60)
        self.logger.info("🚀 수어 인식 모델 학습 초기화 완료")
        self.logger.info("=" * 60)
        self.logger.info(f"📚 어휘 크기: {self.vocab_size}")
        self.logger.info(f"🔢 모델 파라미터: {count_parameters(self.model):,}")
        self.logger.info(f"📊 학습 데이터: {len(self.train_loader.dataset):,}개")
        self.logger.info(f"📊 검증 데이터: {len(self.val_loader.dataset):,}개")
        self.logger.info(f"💾 디바이스: {self.device}")
        self.logger.info(f"🎯 목표 에포크: {num_epochs}")
        self.logger.info(f"📈 학습률: {learning_rate}")
        self.logger.info(f"⚖️ 클래스 가중치: {use_class_weights}")
        self.logger.info("=" * 60)
    
    def _load_checkpoint(self, checkpoint_path: Path):
        """체크포인트 로딩"""
        try:
            checkpoint = load_checkpoint(checkpoint_path, self.model, self.optimizer)
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_accuracy = checkpoint.get('accuracy', 0.0)
            
            # 학습 기록 복원 (있다면)
            if 'train_losses' in checkpoint:
                self.train_losses = checkpoint['train_losses']
                self.val_losses = checkpoint['val_losses']
                self.train_accuracies = checkpoint['train_accuracies']
                self.val_accuracies = checkpoint['val_accuracies']
            
            self.logger.info(f"체크포인트 재개: 에포크 {self.start_epoch}부터 시작")
            
        except Exception as e:
            self.logger.error(f"체크포인트 로딩 실패: {e}")
            self.logger.info("처음부터 학습을 시작합니다.")
    
    def _save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, 
                        val_accuracy: float, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint_data = {
            'epoch': epoch,
            'loss': val_loss,
            'accuracy': val_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'vocab_size': self.vocab_size,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'gradient_clip_val': self.gradient_clip_val
            }
        }
        
        # 기본 체크포인트 저장
        checkpoint_path = CHECKPOINTS_PATH / "last.pt"
        save_checkpoint(
            self.model.model,  # BiLSTMSignClassifier 저장
            self.optimizer,
            epoch,
            val_loss,
            val_accuracy,
            checkpoint_path,
            is_best=is_best
        )
        
        # 정기 체크포인트 저장
        if (epoch + 1) % self.save_every == 0:
            periodic_path = CHECKPOINTS_PATH / f"epoch_{epoch+1:03d}.pt"
            save_checkpoint(
                self.model.model,
                self.optimizer,
                epoch,
                val_loss,
                val_accuracy,
                periodic_path
            )
    
    def _compute_accuracy(self, outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """정확도 계산"""
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        return correct / total * 100.0
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """한 에포크 학습"""
        self.model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(self.train_loader)
        
        # 프로그레스 바
        pbar = tqdm(self.train_loader, desc=f"에포크 {epoch+1}/{self.num_epochs} [학습]")
        
        for batch_idx, batch in enumerate(pbar):
            # 데이터를 디바이스로 이동
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            
            # 순전파
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            
            loss = outputs['loss']
            logits = outputs['logits']
            labels = batch['label']
            
            # 역전파
            loss.backward()
            
            # 기울기 클리핑
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip_val
                )
            
            self.optimizer.step()
            
            # 메트릭 계산
            accuracy = self._compute_accuracy(logits, labels)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            
            # 프로그레스 바 업데이트
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{accuracy:.2f}%",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # TensorBoard 로깅 (배치별)
            global_step = epoch * num_batches + batch_idx
            self.writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
            self.writer.add_scalar('Train/Accuracy_Step', accuracy, global_step)
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        return avg_loss, avg_accuracy
    
    def validate_epoch(self, epoch: int) -> Tuple[float, float, Dict]:
        """한 에포크 검증"""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = len(self.val_loader)
        
        # 클래스별 예측 결과 저장
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"에포크 {epoch+1}/{self.num_epochs} [검증]")
            
            for batch in pbar:
                # 데이터를 디바이스로 이동
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # 순전파
                outputs = self.model(batch)
                
                loss = outputs['loss']
                logits = outputs['logits']
                labels = batch['label']
                
                # 메트릭 계산
                accuracy = self._compute_accuracy(logits, labels)
                
                total_loss += loss.item()
                total_accuracy += accuracy
                
                # 예측 결과 저장
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 프로그레스 바 업데이트
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{accuracy:.2f}%"
                })
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        # 클래스별 통계 계산
        class_stats = self._compute_class_statistics(all_predictions, all_labels)
        
        return avg_loss, avg_accuracy, class_stats
    
    def _compute_class_statistics(self, predictions: List[int], labels: List[int]) -> Dict:
        """클래스별 통계 계산"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        # 전체 정확도
        overall_accuracy = accuracy_score(labels, predictions)
        
        # 클래스별 precision, recall, f1
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # 평균 메트릭
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1)
        
        return {
            'overall_accuracy': overall_accuracy * 100,
            'avg_precision': avg_precision * 100,
            'avg_recall': avg_recall * 100,
            'avg_f1': avg_f1 * 100,
            'class_precision': precision,
            'class_recall': recall,
            'class_f1': f1,
            'class_support': support
        }
    
    def train(self):
        """전체 학습 프로세스"""
        self.logger.info("🎯 학습 시작!")
        start_time = time.time()
        
        try:
            for epoch in range(self.start_epoch, self.num_epochs):
                epoch_start_time = time.time()
                
                # 학습
                train_loss, train_accuracy = self.train_epoch(epoch)
                
                # 검증
                val_loss, val_accuracy, class_stats = self.validate_epoch(epoch)
                
                # 학습률 스케줄러 업데이트
                self.scheduler.step(val_accuracy)
                
                # 메트릭 저장
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accuracies.append(train_accuracy)
                self.val_accuracies.append(val_accuracy)
                
                # 최고 성능 체크
                is_best = val_accuracy > self.best_accuracy
                if is_best:
                    self.best_accuracy = val_accuracy
                
                # 체크포인트 저장
                self._save_checkpoint(epoch, train_loss, val_loss, val_accuracy, is_best)
                
                # TensorBoard 로깅 (에포크별)
                self.writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
                self.writer.add_scalar('Train/Accuracy_Epoch', train_accuracy, epoch)
                self.writer.add_scalar('Val/Loss_Epoch', val_loss, epoch)
                self.writer.add_scalar('Val/Accuracy_Epoch', val_accuracy, epoch)
                self.writer.add_scalar('Val/Precision', class_stats['avg_precision'], epoch)
                self.writer.add_scalar('Val/Recall', class_stats['avg_recall'], epoch)
                self.writer.add_scalar('Val/F1', class_stats['avg_f1'], epoch)
                self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
                
                # 에포크 결과 로깅
                epoch_time = time.time() - epoch_start_time
                
                self.logger.info("=" * 80)
                self.logger.info(f"에포크 {epoch+1}/{self.num_epochs} 완료 ({format_time(epoch_time)})")
                self.logger.info(f"📈 학습   - 손실: {train_loss:.4f}, 정확도: {train_accuracy:.2f}%")
                self.logger.info(f"📊 검증   - 손실: {val_loss:.4f}, 정확도: {val_accuracy:.2f}%")
                self.logger.info(f"🎯 상세   - P: {class_stats['avg_precision']:.2f}%, "
                               f"R: {class_stats['avg_recall']:.2f}%, F1: {class_stats['avg_f1']:.2f}%")
                self.logger.info(f"⭐ 최고   - {self.best_accuracy:.2f}% {'(NEW!)' if is_best else ''}")
                self.logger.info(f"📚 학습률 - {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # 조기 종료 체크 (선택적)
                if self.optimizer.param_groups[0]['lr'] < 1e-7:
                    self.logger.info("학습률이 최소값에 도달하여 학습을 종료합니다.")
                    break
        
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 학습이 중단되었습니다.")
        
        except Exception as e:
            self.logger.error(f"학습 중 오류 발생: {e}")
            raise
        
        finally:
            # 학습 완료 처리
            total_time = time.time() - start_time
            
            self.logger.info("=" * 80)
            self.logger.info("🎉 학습 완료!")
            self.logger.info("=" * 80)
            self.logger.info(f"⏱️ 총 학습 시간: {format_time(total_time)}")
            self.logger.info(f"🏆 최고 검증 정확도: {self.best_accuracy:.2f}%")
            self.logger.info(f"💾 체크포인트 저장 위치: {CHECKPOINTS_PATH}")
            self.logger.info("=" * 80)
            
            # 학습 곡선 저장
            self._save_training_plots()
            
            # TensorBoard 종료
            self.writer.close()
    
    def _save_training_plots(self):
        """학습 곡선 그래프 저장"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            epochs = range(1, len(self.train_losses) + 1)
            
            # 손실 곡선
            ax1.plot(epochs, self.train_losses, 'b-', label='학습 손실', linewidth=2)
            ax1.plot(epochs, self.val_losses, 'r-', label='검증 손실', linewidth=2)
            ax1.set_title('학습/검증 손실', fontsize=14, fontweight='bold')
            ax1.set_xlabel('에포크')
            ax1.set_ylabel('손실')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 정확도 곡선
            ax2.plot(epochs, self.train_accuracies, 'b-', label='학습 정확도', linewidth=2)
            ax2.plot(epochs, self.val_accuracies, 'r-', label='검증 정확도', linewidth=2)
            ax2.set_title('학습/검증 정확도', fontsize=14, fontweight='bold')
            ax2.set_xlabel('에포크')
            ax2.set_ylabel('정확도 (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 과적합 지표 (학습-검증 차이)
            train_val_gap = [t - v for t, v in zip(self.train_accuracies, self.val_accuracies)]
            ax3.plot(epochs, train_val_gap, 'g-', linewidth=2)
            ax3.set_title('과적합 지표 (학습-검증 정확도 차이)', fontsize=14, fontweight='bold')
            ax3.set_xlabel('에포크')
            ax3.set_ylabel('정확도 차이 (%)')
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax3.grid(True, alpha=0.3)
            
            # 최고 성능까지의 진행
            best_epochs = []
            best_vals = []
            current_best = 0
            for i, val_acc in enumerate(self.val_accuracies):
                if val_acc > current_best:
                    current_best = val_acc
                    best_epochs.append(i + 1)
                    best_vals.append(val_acc)
            
            ax4.plot(best_epochs, best_vals, 'ro-', linewidth=2, markersize=6)
            ax4.set_title('최고 성능 진행', fontsize=14, fontweight='bold')
            ax4.set_xlabel('에포크')
            ax4.set_ylabel('최고 검증 정확도 (%)')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 저장
            plot_path = LOGS_PATH / "training_curves.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"📊 학습 곡선 저장: {plot_path}")
            
        except Exception as e:
            self.logger.warning(f"학습 곡선 저장 실패: {e}")

def main():
    """메인 실행 함수"""
    trainer = SignLanguageTrainer(
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        use_class_weights=True,
        gradient_clip_val=1.0,
        save_every=5
    )
    
    trainer.train()
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("🎉 학습이 성공적으로 완료되었습니다!")
    else:
        print("❌ 학습이 실패했습니다.")