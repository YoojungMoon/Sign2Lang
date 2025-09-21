#!/usr/bin/env python3
"""
SU:DA - 수어 인식 모델 평가
학습된 모델의 성능 평가 및 분석
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)
from collections import defaultdict, Counter
import json
from tqdm import tqdm
import time

from utils import (
    setup_logger, load_checkpoint, get_device,
    CHECKPOINTS_PATH, LOGS_PATH, format_time
)
from model import create_model
from dataset import create_data_loaders
from vocab import SignVocabulary

class SignLanguageEvaluator:
    """수어 인식 모델 평가 클래스"""
    
    def __init__(self,
                 checkpoint_path: Optional[Path] = None,
                 vocab_path: Optional[Path] = None,
                 batch_size: int = 32,
                 device: Optional[torch.device] = None,
                 save_predictions: bool = True):
        """
        Args:
            checkpoint_path: 평가할 모델 체크포인트 경로
            vocab_path: 사전 파일 경로
            batch_size: 배치 크기
            device: 평가 디바이스
            save_predictions: 예측 결과 저장 여부
        """
        self.logger = setup_logger("SignLanguageEvaluator", "eval.log")
        self.save_predictions = save_predictions
        
        # 디바이스 설정
        self.device = device if device is not None else get_device()
        
        # 사전 로딩
        self.vocab = SignVocabulary(vocab_path)
        self.vocab_size = len(self.vocab)
        
        # 모델 생성 및 로딩
        self.model = create_model(vocab_size=self.vocab_size, device=self.device)
        
        if checkpoint_path is None:
            checkpoint_path = CHECKPOINTS_PATH / "best.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트 파일이 없습니다: {checkpoint_path}")
        
        self._load_model(checkpoint_path)
        
        # 데이터 로더 생성
        self.train_loader, self.val_loader = create_data_loaders(
            train_batch_size=batch_size,
            val_batch_size=batch_size,
            vocab_path=vocab_path
        )
        
        # 평가 결과 저장용
        self.results = {}
        
        self.logger.info("=" * 60)
        self.logger.info("🔍 수어 인식 모델 평가 초기화 완료")
        self.logger.info("=" * 60)
        self.logger.info(f"📚 어휘 크기: {self.vocab_size}")
        self.logger.info(f"📊 학습 데이터: {len(self.train_loader.dataset):,}개")
        self.logger.info(f"📊 검증 데이터: {len(self.val_loader.dataset):,}개")
        self.logger.info(f"💾 디바이스: {self.device}")
        self.logger.info(f"📂 체크포인트: {checkpoint_path}")
        self.logger.info("=" * 60)
    
    def _load_model(self, checkpoint_path: Path):
        """모델 체크포인트 로딩"""
        try:
            checkpoint = load_checkpoint(checkpoint_path, self.model.model)
            self.model.eval()
            
            self.logger.info(f"✅ 모델 로딩 완료")
            self.logger.info(f"  - 에포크: {checkpoint['epoch']}")
            self.logger.info(f"  - 검증 손실: {checkpoint['loss']:.4f}")
            self.logger.info(f"  - 검증 정확도: {checkpoint['accuracy']:.2f}%")
            
        except Exception as e:
            self.logger.error(f"모델 로딩 실패: {e}")
            raise
    
    def evaluate_dataset(self, data_loader, dataset_name: str) -> Dict:
        """데이터셋 평가"""
        self.logger.info(f"🎯 {dataset_name} 데이터셋 평가 시작")
        
        self.model.eval()
        start_time = time.time()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_word_labels = []
        all_word_ids = []
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc=f"{dataset_name} 평가 중")
            
            for batch in pbar:
                # 데이터를 디바이스로 이동
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # 예측
                outputs = self.model(batch)
                pred_outputs = self.model.predict_batch(batch)
                
                # 결과 수집
                loss = outputs['loss']
                total_loss += loss.item()
                num_batches += 1
                
                predictions = pred_outputs['predictions'].cpu().numpy()
                probabilities = pred_outputs['probabilities'].cpu().numpy()
                labels = batch['label'].cpu().numpy()
                word_labels = batch['word_label']
                word_ids = batch['word_id'].cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                all_probabilities.extend(probabilities)
                all_word_labels.extend(word_labels)
                all_word_ids.extend(word_ids)
                
                # 프로그레스 바 업데이트
                accuracy = accuracy_score(labels, predictions) * 100
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{accuracy:.2f}%"
                })
        
        eval_time = time.time() - start_time
        avg_loss = total_loss / num_batches
        
        # 전체 메트릭 계산
        metrics = self._compute_detailed_metrics(
            all_predictions, all_labels, all_probabilities,
            all_word_labels, all_word_ids
        )
        metrics['loss'] = avg_loss
        metrics['eval_time'] = eval_time
        
        self.logger.info(f"✅ {dataset_name} 평가 완료 ({format_time(eval_time)})")
        self.logger.info(f"  - 전체 정확도: {metrics['accuracy']:.2f}%")
        self.logger.info(f"  - 평균 손실: {avg_loss:.4f}")
        self.logger.info(f"  - Precision: {metrics['macro_precision']:.2f}%")
        self.logger.info(f"  - Recall: {metrics['macro_recall']:.2f}%")
        self.logger.info(f"  - F1-Score: {metrics['macro_f1']:.2f}%")
        
        return metrics
    
    def _compute_detailed_metrics(self, predictions: List[int], labels: List[int],
                                probabilities: np.ndarray, word_labels: List[str],
                                word_ids: List[int]) -> Dict:
        """상세 메트릭 계산"""
        
        # 기본 메트릭
        accuracy = accuracy_score(labels, predictions) * 100
        
        # 클래스별 메트릭
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # 매크로 평균
        macro_precision = np.mean(precision) * 100
        macro_recall = np.mean(recall) * 100
        macro_f1 = np.mean(f1) * 100
        
        # 마이크로 평균 (전체 정확도와 동일)
        micro_precision = accuracy
        micro_recall = accuracy
        micro_f1 = accuracy
        
        # 가중 평균
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        weighted_precision *= 100
        weighted_recall *= 100
        weighted_f1 *= 100
        
        # Top-k 정확도
        top3_accuracy = self._compute_topk_accuracy(probabilities, labels, k=3)
        top5_accuracy = self._compute_topk_accuracy(probabilities, labels, k=5)
        
        # 신뢰도 통계
        max_probs = np.max(probabilities, axis=1)
        confidence_stats = {
            'mean_confidence': np.mean(max_probs) * 100,
            'median_confidence': np.median(max_probs) * 100,
            'std_confidence': np.std(max_probs) * 100,
            'min_confidence': np.min(max_probs) * 100,
            'max_confidence': np.max(max_probs) * 100
        }
        
        # 혼동 행렬
        cm = confusion_matrix(labels, predictions)
        
        # 단어별 성능 분석
        word_performance = self._analyze_word_performance(
            predictions, labels, word_labels, word_ids, probabilities
        )
        
        return {
            # 기본 메트릭
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            
            # Top-k 정확도
            'top3_accuracy': top3_accuracy,
            'top5_accuracy': top5_accuracy,
            
            # 신뢰도 통계
            **confidence_stats,
            
            # 상세 데이터
            'class_precision': precision,
            'class_recall': recall,
            'class_f1': f1,
            'class_support': support,
            'confusion_matrix': cm,
            'word_performance': word_performance,
            
            # 원본 데이터 (저장용)
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities,
            'word_labels': word_labels,
            'word_ids': word_ids
        }
    
    def _compute_topk_accuracy(self, probabilities: np.ndarray, labels: List[int], k: int) -> float:
        """Top-k 정확도 계산"""
        top_k_indices = np.argsort(probabilities, axis=1)[:, -k:]
        correct = 0
        
        for i, true_label in enumerate(labels):
            if true_label in top_k_indices[i]:
                correct += 1
        
        return (correct / len(labels)) * 100
    
    def _analyze_word_performance(self, predictions: List[int], labels: List[int],
                                word_labels: List[str], word_ids: List[int],
                                probabilities: np.ndarray) -> Dict:
        """단어별 성능 분석"""
        
        word_stats = defaultdict(lambda: {
            'correct': 0, 'total': 0, 'confidences': []
        })
        
        for pred, label, word_label, word_id, prob in zip(
            predictions, labels, word_labels, word_ids, probabilities
        ):
            word_stats[word_label]['total'] += 1
            word_stats[word_label]['confidences'].append(np.max(prob))
            
            if pred == label:
                word_stats[word_label]['correct'] += 1
        
        # 단어별 정확도 계산
        word_accuracies = {}
        for word, stats in word_stats.items():
            accuracy = (stats['correct'] / stats['total']) * 100
            avg_confidence = np.mean(stats['confidences']) * 100
            
            word_accuracies[word] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total'],
                'avg_confidence': avg_confidence
            }
        
        # 성능별 단어 분류
        best_words = sorted(word_accuracies.items(), 
                           key=lambda x: x[1]['accuracy'], reverse=True)[:10]
        worst_words = sorted(word_accuracies.items(), 
                            key=lambda x: x[1]['accuracy'])[:10]
        
        return {
            'word_accuracies': word_accuracies,
            'best_words': best_words,
            'worst_words': worst_words,
            'avg_word_accuracy': np.mean([w['accuracy'] for w in word_accuracies.values()])
        }
    
    def save_detailed_results(self, train_metrics: Dict, val_metrics: Dict):
        """상세 결과 저장"""
        if not self.save_predictions:
            return
        
        results_dir = LOGS_PATH / "evaluation_results"
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 1. 전체 결과 요약 저장
        summary = {
            'timestamp': timestamp,
            'train_metrics': {k: v for k, v in train_metrics.items() 
                            if not isinstance(v, (np.ndarray, list))},
            'val_metrics': {k: v for k, v in val_metrics.items() 
                          if not isinstance(v, (np.ndarray, list))},
            'vocab_size': self.vocab_size
        }
        
        summary_file = results_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 2. 예측 결과 저장
        self._save_predictions_csv(val_metrics, results_dir, timestamp)
        
        # 3. 시각화 저장
        self._save_visualizations(train_metrics, val_metrics, results_dir, timestamp)
        
        # 4. 분류 리포트 저장
        self._save_classification_report(val_metrics, results_dir, timestamp)
        
        self.logger.info(f"📊 평가 결과 저장 완료: {results_dir}")
    
    def _save_predictions_csv(self, metrics: Dict, results_dir: Path, timestamp: str):
        """예측 결과를 CSV로 저장"""
        
        predictions_data = []
        for i, (pred, label, word_label, word_id, prob) in enumerate(zip(
            metrics['predictions'], metrics['labels'], 
            metrics['word_labels'], metrics['word_ids'], metrics['probabilities']
        )):
            
            pred_word = self.vocab.index_to_word(pred)
            true_word = self.vocab.index_to_word(label)
            max_prob = np.max(prob)
            
            predictions_data.append({
                'sample_id': i,
                'word_id': word_id,
                'true_label_idx': label,
                'pred_label_idx': pred,
                'true_word': true_word,
                'pred_word': pred_word,
                'word_label': word_label,
                'max_probability': max_prob,
                'is_correct': pred == label
            })
        
        df = pd.DataFrame(predictions_data)
        predictions_file = results_dir / f"predictions_{timestamp}.csv"
        df.to_csv(predictions_file, index=False, encoding='utf-8')
        
        self.logger.info(f"💾 예측 결과 저장: {predictions_file}")
    
    def _save_visualizations(self, train_metrics: Dict, val_metrics: Dict, 
                           results_dir: Path, timestamp: str):
        """시각화 저장"""
        
        # 1. 혼동 행렬 (상위 20개 클래스만)
        self._plot_confusion_matrix(val_metrics['confusion_matrix'], 
                                   results_dir, timestamp)
        
        # 2. 클래스별 성능
        self._plot_class_performance(val_metrics, results_dir, timestamp)
        
        # 3. 신뢰도 분포
        self._plot_confidence_distribution(val_metrics, results_dir, timestamp)
        
        # 4. 단어별 성능
        self._plot_word_performance(val_metrics, results_dir, timestamp)
    
    def _plot_confusion_matrix(self, cm: np.ndarray, results_dir: Path, timestamp: str):
        """혼동 행렬 시각화"""
        
        # 상위 20개 클래스만 선택 (가장 많이 예측된 클래스들)
        top_classes = np.argsort(np.sum(cm, axis=1))[-20:]
        cm_subset = cm[np.ix_(top_classes, top_classes)]
        
        # 클래스 이름 가져오기
        class_names = [self.vocab.index_to_word(i) for i in top_classes]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('혼동 행렬 (상위 20개 클래스)', fontsize=14, fontweight='bold')
        plt.xlabel('예측 라벨')
        plt.ylabel('실제 라벨')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_file = results_dir / f"confusion_matrix_{timestamp}.png"
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_performance(self, metrics: Dict, results_dir: Path, timestamp: str):
        """클래스별 성능 시각화"""
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Precision, Recall, F1 분포
        ax1.hist(metrics['class_precision'], bins=30, alpha=0.7, label='Precision')
        ax1.hist(metrics['class_recall'], bins=30, alpha=0.7, label='Recall')
        ax1.hist(metrics['class_f1'], bins=30, alpha=0.7, label='F1-Score')
        ax1.set_xlabel('점수')
        ax1.set_ylabel('클래스 수')
        ax1.set_title('클래스별 성능 분포')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Support vs Performance
        ax2.scatter(metrics['class_support'], metrics['class_f1'], alpha=0.6)
        ax2.set_xlabel('Support (샘플 수)')
        ax2.set_ylabel('F1-Score')
        ax2.set_title('샘플 수 vs F1-Score')
        ax2.grid(True, alpha=0.3)
        
        # Top/Bottom 성능 클래스들
        sorted_f1 = sorted(enumerate(metrics['class_f1']), key=lambda x: x[1])
        bottom_10 = sorted_f1[:10]
        top_10 = sorted_f1[-10:]
        
        bottom_names = [self.vocab.index_to_word(i) for i, _ in bottom_10]
        bottom_scores = [score for _, score in bottom_10]
        top_names = [self.vocab.index_to_word(i) for i, _ in top_10]
        top_scores = [score for _, score in top_10]
        
        y_pos = np.arange(10)
        ax3.barh(y_pos, bottom_scores, alpha=0.7, color='red', label='Bottom 10')
        ax3.barh(y_pos + 11, top_scores, alpha=0.7, color='green', label='Top 10')
        ax3.set_yticks(list(y_pos) + list(y_pos + 11))
        ax3.set_yticklabels(bottom_names + top_names, fontsize=8)
        ax3.set_xlabel('F1-Score')
        ax3.set_title('최고/최악 성능 클래스')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        performance_file = results_dir / f"class_performance_{timestamp}.png"
        plt.savefig(performance_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self, metrics: Dict, results_dir: Path, timestamp: str):
        """신뢰도 분포 시각화"""
        
        max_probs = np.max(metrics['probabilities'], axis=1)
        correct_mask = np.array(metrics['predictions']) == np.array(metrics['labels'])
        
        plt.figure(figsize=(12, 8))
        
        # 정답/오답별 신뢰도 분포
        plt.subplot(2, 2, 1)
        plt.hist(max_probs[correct_mask], bins=30, alpha=0.7, label='정답', color='green')
        plt.hist(max_probs[~correct_mask], bins=30, alpha=0.7, label='오답', color='red')
        plt.xlabel('최대 확률')
        plt.ylabel('빈도')
        plt.title('예측 신뢰도 분포')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 신뢰도 vs 정확도
        plt.subplot(2, 2, 2)
        confidence_bins = np.linspace(0, 1, 11)
        accuracies = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i+1])
            if mask.sum() > 0:
                accuracy = correct_mask[mask].mean()
                accuracies.append(accuracy)
            else:
                accuracies.append(0)
        
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        plt.plot(bin_centers, accuracies, 'bo-', linewidth=2, markersize=6)
        plt.xlabel('신뢰도 구간')
        plt.ylabel('정확도')
        plt.title('신뢰도별 정확도')
        plt.grid(True, alpha=0.3)
        
        # 보정 곡선 (Calibration Curve)
        plt.subplot(2, 2, 3)
        from sklearn.calibration import calibration_curve
        fraction_positives, mean_predicted_value = calibration_curve(
            correct_mask, max_probs, n_bins=10
        )
        plt.plot(mean_predicted_value, fraction_positives, 'bo-', label='모델')
        plt.plot([0, 1], [0, 1], 'k--', label='완벽한 보정')
        plt.xlabel('평균 예측 확률')
        plt.ylabel('실제 정답 비율')
        plt.title('보정 곡선')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 신뢰도 통계
        plt.subplot(2, 2, 4)
        stats_text = f"""
        평균 신뢰도: {metrics['mean_confidence']:.2f}%
        중앙값 신뢰도: {metrics['median_confidence']:.2f}%
        표준편차: {metrics['std_confidence']:.2f}%
        최소값: {metrics['min_confidence']:.2f}%
        최대값: {metrics['max_confidence']:.2f}%
        
        정답 평균 신뢰도: {max_probs[correct_mask].mean()*100:.2f}%
        오답 평균 신뢰도: {max_probs[~correct_mask].mean()*100:.2f}%
        """
        plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        plt.axis('off')
        plt.title('신뢰도 통계')
        
        plt.tight_layout()
        confidence_file = results_dir / f"confidence_analysis_{timestamp}.png"
        plt.savefig(confidence_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_word_performance(self, metrics: Dict, results_dir: Path, timestamp: str):
        """단어별 성능 시각화"""
        
        word_perf = metrics['word_performance']
        
        plt.figure(figsize=(15, 10))
        
        # 최고 성능 단어들
        plt.subplot(2, 2, 1)
        best_words = word_perf['best_words'][:10]
        words = [w[0] for w in best_words]
        accs = [w[1]['accuracy'] for w in best_words]
        
        plt.barh(range(len(words)), accs, color='green', alpha=0.7)
        plt.yticks(range(len(words)), words)
        plt.xlabel('정확도 (%)')
        plt.title('최고 성능 단어 (Top 10)')
        plt.grid(True, alpha=0.3)
        
        # 최악 성능 단어들
        plt.subplot(2, 2, 2)
        worst_words = word_perf['worst_words'][:10]
        words = [w[0] for w in worst_words]
        accs = [w[1]['accuracy'] for w in worst_words]
        
        plt.barh(range(len(words)), accs, color='red', alpha=0.7)
        plt.yticks(range(len(words)), words)
        plt.xlabel('정확도 (%)')
        plt.title('최악 성능 단어 (Bottom 10)')
        plt.grid(True, alpha=0.3)
        
        # 단어별 정확도 분포
        plt.subplot(2, 2, 3)
        all_accs = [w['accuracy'] for w in word_perf['word_accuracies'].values()]
        plt.hist(all_accs, bins=30, alpha=0.7, color='blue')
        plt.axvline(word_perf['avg_word_accuracy'], color='red', linestyle='--', 
                   label=f'평균: {word_perf["avg_word_accuracy"]:.1f}%')
        plt.xlabel('정확도 (%)')
        plt.ylabel('단어 수')
        plt.title('단어별 정확도 분포')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 샘플 수 vs 정확도
        plt.subplot(2, 2, 4)
        totals = [w['total'] for w in word_perf['word_accuracies'].values()]
        accs = [w['accuracy'] for w in word_perf['word_accuracies'].values()]
        
        plt.scatter(totals, accs, alpha=0.6)
        plt.xlabel('샘플 수')
        plt.ylabel('정확도 (%)')
        plt.title('샘플 수 vs 정확도')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        word_perf_file = results_dir / f"word_performance_{timestamp}.png"
        plt.savefig(word_perf_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_classification_report(self, metrics: Dict, results_dir: Path, timestamp: str):
        """분류 리포트 저장"""
        
        # scikit-learn 분류 리포트
        target_names = [self.vocab.index_to_word(i) for i in range(self.vocab_size)]
        report = classification_report(
            metrics['labels'], metrics['predictions'],
            target_names=target_names, output_dict=True, zero_division=0
        )
        
        # JSON으로 저장
        report_file = results_dir / f"classification_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 텍스트 형태로도 저장
        text_report = classification_report(
            metrics['labels'], metrics['predictions'],
            target_names=target_names, zero_division=0
        )
        
        text_file = results_dir / f"classification_report_{timestamp}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        self.logger.info(f"📋 분류 리포트 저장: {report_file}")
    
    def run_full_evaluation(self) -> Dict:
        """전체 평가 실행"""
        self.logger.info("🚀 전체 평가 시작!")
        
        start_time = time.time()
        
        # 학습 데이터 평가
        train_metrics = self.evaluate_dataset(self.train_loader, "학습")
        
        # 검증 데이터 평가
        val_metrics = self.evaluate_dataset(self.val_loader, "검증")
        
        # 결과 저장
        self.save_detailed_results(train_metrics, val_metrics)
        
        # 요약 출력
        total_time = time.time() - start_time
        
        self.logger.info("=" * 80)
        self.logger.info("🎉 전체 평가 완료!")
        self.logger.info("=" * 80)
        self.logger.info(f"⏱️  총 평가 시간: {format_time(total_time)}")
        self.logger.info(f"📊 학습 데이터 정확도: {train_metrics['accuracy']:.2f}%")
        self.logger.info(f"📊 검증 데이터 정확도: {val_metrics['accuracy']:.2f}%")
        self.logger.info(f"🎯 Top-3 정확도: {val_metrics['top3_accuracy']:.2f}%")
        self.logger.info(f"🎯 Top-5 정확도: {val_metrics['top5_accuracy']:.2f}%")
        self.logger.info(f"📈 Macro F1-Score: {val_metrics['macro_f1']:.2f}%")
        self.logger.info(f"📈 Weighted F1-Score: {val_metrics['weighted_f1']:.2f}%")
        self.logger.info(f"🔒 평균 신뢰도: {val_metrics['mean_confidence']:.2f}%")
        self.logger.info(f"💾 결과 저장 위치: {LOGS_PATH / 'evaluation_results'}")
        self.logger.info("=" * 80)
        
        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'evaluation_time': total_time
        }
    
    def quick_evaluation(self, num_samples: int = 100) -> Dict:
        """빠른 평가 (일부 샘플만)"""
        self.logger.info(f"⚡ 빠른 평가 시작 ({num_samples}개 샘플)")
        
        self.model.eval()
        
        # 검증 데이터에서 일부만 추출
        sample_indices = np.random.choice(
            len(self.val_loader.dataset), 
            min(num_samples, len(self.val_loader.dataset)), 
            replace=False
        )
        
        predictions = []
        labels = []
        confidences = []
        
        with torch.no_grad():
            for idx in tqdm(sample_indices, desc="빠른 평가"):
                sample = self.val_loader.dataset[idx]
                
                # 배치 차원 추가
                batch = {key: tensor.unsqueeze(0).to(self.device) 
                        for key, tensor in sample.items() 
                        if isinstance(tensor, torch.Tensor)}
                batch['word_label'] = [sample['word_label']]
                
                # 예측
                pred_outputs = self.model.predict_batch(batch)
                
                predictions.append(pred_outputs['predictions'][0].item())
                labels.append(sample['label'].item())
                confidences.append(pred_outputs['max_probabilities'][0].item())
        
        # 기본 메트릭 계산
        accuracy = accuracy_score(labels, predictions) * 100
        avg_confidence = np.mean(confidences) * 100
        
        correct_mask = np.array(predictions) == np.array(labels)
        correct_confidence = np.mean(np.array(confidences)[correct_mask]) * 100
        wrong_confidence = np.mean(np.array(confidences)[~correct_mask]) * 100
        
        self.logger.info(f"✅ 빠른 평가 완료:")
        self.logger.info(f"  - 정확도: {accuracy:.2f}%")
        self.logger.info(f"  - 평균 신뢰도: {avg_confidence:.2f}%")
        self.logger.info(f"  - 정답 신뢰도: {correct_confidence:.2f}%")
        self.logger.info(f"  - 오답 신뢰도: {wrong_confidence:.2f}%")
        
        return {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'correct_confidence': correct_confidence,
            'wrong_confidence': wrong_confidence,
            'num_samples': len(sample_indices)
        }
    
    def evaluate_single_sample(self, sample_idx: int) -> Dict:
        """단일 샘플 상세 분석"""
        if sample_idx >= len(self.val_loader.dataset):
            raise IndexError(f"샘플 인덱스 범위 초과: {sample_idx}")
        
        sample = self.val_loader.dataset[sample_idx]
        
        # 배치 차원 추가
        batch = {key: tensor.unsqueeze(0).to(self.device) 
                for key, tensor in sample.items() 
                if isinstance(tensor, torch.Tensor)}
        batch['word_label'] = [sample['word_label']]
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch)
            pred_outputs = self.model.predict_batch(batch)
        
        # 상위 5개 예측 결과
        probs = pred_outputs['probabilities'][0].cpu().numpy()
        top5_indices = np.argsort(probs)[-5:][::-1]
        top5_words = [self.vocab.index_to_word(idx) for idx in top5_indices]
        top5_probs = [probs[idx] * 100 for idx in top5_indices]
        
        result = {
            'sample_info': {
                'index': sample_idx,
                'word_id': sample['word_id'].item(),
                'true_word': sample['word_label'],
                'true_label_idx': sample['label'].item(),
                'original_length': sample['original_length'].item()
            },
            'prediction': {
                'predicted_idx': pred_outputs['predictions'][0].item(),
                'predicted_word': self.vocab.index_to_word(pred_outputs['predictions'][0].item()),
                'confidence': pred_outputs['max_probabilities'][0].item() * 100,
                'is_correct': pred_outputs['predictions'][0].item() == sample['label'].item()
            },
            'top5_predictions': {
                'words': top5_words,
                'probabilities': top5_probs,
                'indices': top5_indices.tolist()
            },
            'loss': outputs['loss'].item()
        }
        
        self.logger.info(f"🔍 샘플 {sample_idx} 분석:")
        self.logger.info(f"  실제: {result['sample_info']['true_word']}")
        self.logger.info(f"  예측: {result['prediction']['predicted_word']}")
        self.logger.info(f"  신뢰도: {result['prediction']['confidence']:.2f}%")
        self.logger.info(f"  정답 여부: {'✅' if result['prediction']['is_correct'] else '❌'}")
        self.logger.info(f"  Top-5: {', '.join([f'{w}({p:.1f}%)' for w, p in zip(top5_words, top5_probs)])}")
        
        return result

def main():
    """메인 실행 함수"""
    
    # 기본 평가
    evaluator = SignLanguageEvaluator(
        checkpoint_path=CHECKPOINTS_PATH / "best.pt",
        batch_size=32,
        save_predictions=True
    )
    
    # 전체 평가 실행
    results = evaluator.run_full_evaluation()
    
    # 빠른 평가도 실행
    quick_results = evaluator.quick_evaluation(num_samples=200)
    
    # 몇 개 샘플 상세 분석
    print("\n🔍 샘플별 상세 분석:")
    for i in range(min(5, len(evaluator.val_loader.dataset))):
        evaluator.evaluate_single_sample(i)
        print("-" * 40)
    
    print("\n🎉 평가가 성공적으로 완료되었습니다!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ 평가 완료!")
    else:
        print("❌ 평가 실패!")