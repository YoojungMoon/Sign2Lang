#!/usr/bin/env python3
"""
SU:DA - 수어 인식 모델
BiLSTM 기반 시퀀스 분류 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

from utils import (
    setup_logger, KEYPOINT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
    TOTAL_LANDMARKS, count_parameters
)

class PositionalEncoding(nn.Module):
    """위치 인코딩 (선택적 사용)"""
    
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, d_model]
        Returns:
            위치 인코딩이 추가된 텐서
        """
        return x + self.pe[:x.size(0), :]

class AttentionLayer(nn.Module):
    """어텐션 레이어"""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=False
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch_size, hidden_dim]
            mask: [batch_size, seq_len] 패딩 마스크
        Returns:
            어텐션 적용된 텐서
        """
        # Self-attention
        if mask is not None:
            # mask를 attention용으로 변환 [batch_size, seq_len] -> [batch_size, seq_len]
            # 패딩된 부분은 True로 설정 (attention에서 무시됨)
            key_padding_mask = (mask == 0)  # [batch_size, seq_len]
        else:
            key_padding_mask = None
        
        attn_output, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        
        # Residual connection + Layer norm
        x = self.norm(x + self.dropout(attn_output))
        
        return x

class KeypointEmbedding(nn.Module):
    """키포인트 임베딩 레이어"""
    
    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # 키포인트 특징 추출
        self.projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 신체 부위별 가중치 (선택적)
        self.body_part_weights = nn.Parameter(torch.ones(4))  # pose, hand_l, hand_r, face
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim] 키포인트 시퀀스
        Returns:
            [batch_size, seq_len, embed_dim] 임베딩된 시퀀스
        """
        batch_size, seq_len, _ = x.shape
        
        # 신체 부위별 가중치 적용 (선택적)
        if self.input_dim == TOTAL_LANDMARKS * 3:  # 1629 = 543 * 3
            # pose(33*3), hand_left(21*3), hand_right(21*3), face(468*3)
            pose_end = 33 * 3
            hand_l_end = pose_end + 21 * 3
            hand_r_end = hand_l_end + 21 * 3
            face_end = hand_r_end + 468 * 3
            
            x_weighted = x.clone()
            x_weighted[:, :, :pose_end] *= self.body_part_weights[0]
            x_weighted[:, :, pose_end:hand_l_end] *= self.body_part_weights[1]
            x_weighted[:, :, hand_l_end:hand_r_end] *= self.body_part_weights[2]
            x_weighted[:, :, hand_r_end:face_end] *= self.body_part_weights[3]
            x = x_weighted
        
        # 임베딩 투영
        embedded = self.projection(x)  # [batch_size, seq_len, embed_dim]
        
        return embedded

class BiLSTMSignClassifier(nn.Module):
    """BiLSTM 기반 수어 분류 모델"""
    
    def __init__(self,
                 input_dim: int = TOTAL_LANDMARKS * 3,  # 1629
                 hidden_dim: int = HIDDEN_DIM,           # 256
                 num_layers: int = NUM_LAYERS,           # 2
                 num_classes: int = 1500,                # 어휘 크기
                 dropout: float = DROPOUT,               # 0.3
                 use_attention: bool = True,
                 use_positional_encoding: bool = False):
        """
        Args:
            input_dim: 입력 키포인트 차원 (기본 1629)
            hidden_dim: LSTM 숨김층 차원
            num_layers: LSTM 레이어 수
            num_classes: 분류할 클래스 수 (어휘 크기)
            dropout: 드롭아웃 비율
            use_attention: 어텐션 사용 여부
            use_positional_encoding: 위치 인코딩 사용 여부
        """
        super().__init__()
        
        self.logger = setup_logger("BiLSTMSignClassifier")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_attention = use_attention
        self.use_positional_encoding = use_positional_encoding
        
        # 1. 키포인트 임베딩
        self.keypoint_embedding = KeypointEmbedding(
            input_dim=input_dim,
            embed_dim=hidden_dim,
            dropout=dropout
        )
        
        # 2. 위치 인코딩 (선택적)
        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # 3. BiLSTM 레이어들
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = hidden_dim
            
            lstm_layer = nn.LSTM(
                input_size=layer_input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0 if i == num_layers - 1 else dropout,
                bidirectional=True
            )
            self.lstm_layers.append(lstm_layer)
        
        # BiLSTM 출력 차원 (양방향이므로 2배)
        lstm_output_dim = hidden_dim * 2
        
        # 4. 어텐션 레이어 (선택적)
        if self.use_attention:
            self.attention = AttentionLayer(lstm_output_dim, dropout)
        
        # 5. 분류 헤드
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # 6. 가중치 초기화
        self._initialize_weights()
        
        # 7. 모델 정보 로깅
        total_params = count_parameters(self)
        self.logger.info(f"모델 초기화 완료:")
        self.logger.info(f"  - 입력 차원: {input_dim}")
        self.logger.info(f"  - 숨김층 차원: {hidden_dim}")
        self.logger.info(f"  - LSTM 레이어 수: {num_layers}")
        self.logger.info(f"  - 출력 클래스 수: {num_classes}")
        self.logger.info(f"  - 어텐션 사용: {use_attention}")
        self.logger.info(f"  - 위치 인코딩 사용: {use_positional_encoding}")
        self.logger.info(f"  - 총 파라미터 수: {total_params:,}")
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM 가중치는 Xavier uniform 초기화
                    nn.init.xavier_uniform_(param)
                elif 'linear' in name or 'projection' in name:
                    # Linear 레이어는 Kaiming 초기화
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, input_dim] 키포인트 시퀀스
            mask: [batch_size, seq_len] 패딩 마스크 (1=실제 데이터, 0=패딩)
            return_features: 중간 특징도 반환할지 여부
        
        Returns:
            딕셔너리 형태의 출력:
            - 'logits': [batch_size, num_classes] 분류 점수
            - 'features': [batch_size, lstm_output_dim] 특징 벡터 (선택적)
            - 'attention_weights': 어텐션 가중치 (선택적)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 키포인트 임베딩
        embedded = self.keypoint_embedding(x)  # [batch_size, seq_len, hidden_dim]
        
        # 2. 위치 인코딩 (선택적)
        if self.use_positional_encoding:
            # LSTM은 batch_first=True이므로 transpose 필요
            embedded = embedded.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
            embedded = self.pos_encoding(embedded)
            embedded = embedded.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        # 3. BiLSTM 레이어들
        lstm_output = embedded
        
        for i, lstm_layer in enumerate(self.lstm_layers):
            # 패킹 (효율적인 패딩 처리)
            if mask is not None:
                lengths = mask.sum(dim=1).cpu()  # 실제 시퀀스 길이들
                packed_input = nn.utils.rnn.pack_padded_sequence(
                    lstm_output, lengths, batch_first=True, enforce_sorted=False
                )
                packed_output, _ = lstm_layer(packed_input)
                lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_output, batch_first=True
                )
            else:
                lstm_output, _ = lstm_layer(lstm_output)
            
            # 중간 레이어에 대해서는 잔차 연결 (선택적)
            if i > 0 and lstm_output.size(-1) == embedded.size(-1):
                lstm_output = lstm_output + embedded
        
        # [batch_size, seq_len, hidden_dim * 2]
        
        # 4. 어텐션 (선택적)
        if self.use_attention:
            # 어텐션을 위해 transpose
            lstm_output_T = lstm_output.transpose(0, 1)  # [seq_len, batch_size, hidden_dim*2]
            attended_output = self.attention(lstm_output_T, mask)
            lstm_output = attended_output.transpose(0, 1)  # [batch_size, seq_len, hidden_dim*2]
        
        # 5. 전역 풀링 (시퀀스 → 고정 크기 벡터)
        if mask is not None:
            # 마스킹된 평균 풀링
            mask_expanded = mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
            masked_output = lstm_output * mask_expanded
            sequence_lengths = mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]
            pooled_output = masked_output.sum(dim=1) / (sequence_lengths + 1e-8)  # [batch_size, hidden_dim*2]
        else:
            # 단순 평균 풀링
            pooled_output = lstm_output.mean(dim=1)  # [batch_size, hidden_dim*2]
        
        # 6. 분류
        logits = self.classifier(pooled_output)  # [batch_size, num_classes]
        
        # 7. 결과 구성
        outputs = {'logits': logits}
        
        if return_features:
            outputs['features'] = pooled_output
        
        return outputs
    
    def predict(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """추론용 함수 (확률값 반환)"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, mask)
            probabilities = F.softmax(outputs['logits'], dim=-1)
        return probabilities
    
    def get_predictions(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """예측 클래스와 확률 반환"""
        probabilities = self.predict(x, mask)
        predicted_classes = torch.argmax(probabilities, dim=-1)
        max_probabilities = torch.max(probabilities, dim=-1)[0]
        
        return predicted_classes, max_probabilities

class SignLanguageModel(nn.Module):
    """전체 수어 인식 모델 래퍼"""
    
    def __init__(self, vocab_size: int, **model_kwargs):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.model = BiLSTMSignClassifier(num_classes=vocab_size, **model_kwargs)
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """배치 데이터로부터 예측"""
        sequences = batch['sequence']          # [batch_size, seq_len, input_dim]
        masks = batch.get('sequence_mask')     # [batch_size, seq_len]
        
        outputs = self.model(sequences, masks)
        
        # 라벨이 있는 경우 손실 계산
        if 'label' in batch:
            labels = batch['label']  # [batch_size]
            loss = F.cross_entropy(outputs['logits'], labels)
            outputs['loss'] = loss
        
        return outputs
    
    def predict_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """배치 예측 (확률값 포함)"""
        self.eval()
        with torch.no_grad():
            sequences = batch['sequence']
            masks = batch.get('sequence_mask')
            
            outputs = self.model(sequences, masks, return_features=True)
            probabilities = F.softmax(outputs['logits'], dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)
            max_probabilities = torch.max(probabilities, dim=-1)[0]
            
            return {
                'predictions': predicted_classes,
                'probabilities': probabilities,
                'max_probabilities': max_probabilities,
                'features': outputs.get('features')
            }

def create_model(vocab_size: int, device: Optional[torch.device] = None, **kwargs) -> SignLanguageModel:
    """모델 생성 헬퍼 함수"""
    model = SignLanguageModel(vocab_size=vocab_size, **kwargs)
    
    if device is not None:
        model = model.to(device)
    
    return model

def main():
    """테스트용 메인 함수"""
    print("🧪 수어 인식 모델 테스트")
    
    # 모델 생성
    vocab_size = 1504  # 1500 단어 + 4 특수토큰
    model = create_model(vocab_size=vocab_size)
    
    print(f"모델 생성 완료: {count_parameters(model):,} 파라미터")
    
    # 더미 데이터 테스트
    batch_size = 8
    seq_len = 200
    input_dim = TOTAL_LANDMARKS * 3  # 1629
    
    # 더미 배치 생성
    dummy_batch = {
        'sequence': torch.randn(batch_size, seq_len, input_dim),
        'sequence_mask': torch.ones(batch_size, seq_len),
        'label': torch.randint(0, vocab_size, (batch_size,))
    }
    
    # 순전파 테스트
    print("\n🔄 순전파 테스트:")
    model.train()
    outputs = model(dummy_batch)
    
    print(f"  로짓 크기: {outputs['logits'].shape}")
    print(f"  손실값: {outputs['loss'].item():.4f}")
    
    # 예측 테스트
    print("\n🎯 예측 테스트:")
    pred_outputs = model.predict_batch(dummy_batch)
    
    print(f"  예측 클래스: {pred_outputs['predictions'][:5]}")
    print(f"  최대 확률: {pred_outputs['max_probabilities'][:5]}")
    print(f"  특징 크기: {pred_outputs['features'].shape}")
    
    print("\n✅ 모델 테스트 완료!")

if __name__ == "__main__":
    main()