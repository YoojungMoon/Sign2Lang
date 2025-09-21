#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SU:DA - 수어 인식 모델
BiLSTM 기반 시퀀스 분류 모델 (안전 초기화 적용판)
"""

from typing import Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    setup_logger, KEYPOINT_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
    TOTAL_LANDMARKS, count_parameters
)

# =========================
# Positional Encoding
# =========================
class PositionalEncoding(nn.Module):
    """위치 인코딩 (선택적 사용)"""
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [max_len, 1, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [seq_len, batch, d_model]
        """
        return x + self.pe[:x.size(0), :]


# =========================
# Attention Layer
# =========================
class AttentionLayer(nn.Module):
    """Multi-Head Self-Attention + Residual + LayerNorm"""
    def __init__(self, model_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=8, dropout=dropout, batch_first=False
        )
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:    [seq_len, batch, model_dim]
        mask: [batch, seq_len] (1=valid, 0=pad)
        """
        key_padding_mask = (mask == 0) if mask is not None else None
        out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        return self.norm(x + self.dropout(out))


# =========================
# Keypoint Embedding
# =========================
class KeypointEmbedding(nn.Module):
    """키포인트 임베딩 레이어"""
    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)  # <- 1D 파라미터 (안전 초기화 필요)
        )

        # 신체 부위별 가중치(선택)
        self.body_part_weights = nn.Parameter(torch.ones(4))  # pose, hand_l, hand_r, face

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, input_dim]
        return: [batch, seq_len, embed_dim]
        """
        if self.input_dim == TOTAL_LANDMARKS * 3:
            # pose(33*3), hand_l(21*3), hand_r(21*3), face(468*3)
            pose_end = 33 * 3
            hand_l_end = pose_end + 21 * 3
            hand_r_end = hand_l_end + 21 * 3
            face_end = hand_r_end + 468 * 3

            x = x.clone()
            x[:, :, :pose_end] *= self.body_part_weights[0]
            x[:, :, pose_end:hand_l_end] *= self.body_part_weights[1]
            x[:, :, hand_l_end:hand_r_end] *= self.body_part_weights[2]
            x[:, :, hand_r_end:face_end] *= self.body_part_weights[3]

        return self.projection(x)


# =========================
# BiLSTM Classifier
# =========================
class BiLSTMSignClassifier(nn.Module):
    """BiLSTM 기반 수어 분류 모델"""
    def __init__(self,
                 input_dim: int = TOTAL_LANDMARKS * 3,   # 1629
                 hidden_dim: int = HIDDEN_DIM,           # 256
                 num_layers: int = NUM_LAYERS,           # 2
                 num_classes: int = 1500,                # 어휘 크기
                 dropout: float = DROPOUT,               # 0.3
                 use_attention: bool = True,
                 use_positional_encoding: bool = False):
        super().__init__()

        self.logger = setup_logger("BiLSTMSignClassifier")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_attention = use_attention
        self.use_positional_encoding = use_positional_encoding

        # 1) 키포인트 임베딩
        self.keypoint_embedding = KeypointEmbedding(
            input_dim=input_dim, embed_dim=hidden_dim, dropout=dropout
        )

        # 2) 위치 인코딩(옵션)
        if self.use_positional_encoding:
            self.pos_encoding = PositionalEncoding(hidden_dim)

        # 3) BiLSTM 스택 (한 층짜리 LSTM을 num_layers번 스택)
        #    내부 LSTM의 dropout은 num_layers==1일 때 무의미하므로 경고 방지 차원에서 0.0으로 고정
        self.lstm_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.lstm_layers.append(nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0.0,           # <-- 경고 제거
                bidirectional=True
            ))

        lstm_output_dim = hidden_dim * 2  # 양방향

        # 4) 어텐션(옵션)
        if self.use_attention:
            self.attention = AttentionLayer(lstm_output_dim, dropout)

        # 5) 분류기
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

        # 6) 안전 가중치 초기화 (모듈 타입 기반)
        self._initialize_weights()

        # 7) 로깅
        self.logger.info("모델 초기화 완료:")
        self.logger.info(f"  - 입력 차원: {input_dim}")
        self.logger.info(f"  - 숨김층 차원: {hidden_dim}")
        self.logger.info(f"  - LSTM 레이어 수: {num_layers}")
        self.logger.info(f"  - 출력 클래스 수: {num_classes}")
        self.logger.info(f"  - 어텐션 사용: {use_attention}")
        self.logger.info(f"  - 위치 인코딩 사용: {use_positional_encoding}")
        self.logger.info(f"  - 총 파라미터 수: {count_parameters(self):,}")

    # -------- 안전 초기화 (권장안) --------
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Linear: 2D weight → fan-in/out 가능, bias는 1D → zeros
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                # LayerNorm: 1D 파라미터 → fan-in/out 사용 금지
                if m.elementwise_affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LSTM):
                # LSTM은 파라미터별로 안전 초기화
                for name, p in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(p)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(p)
                    elif 'bias' in name:
                        nn.init.zeros_(p)
                        # forget gate bias boost (선택)
                        n = p.size(0)
                        p.data[n // 4:n // 2].fill_(1.0)

    # -------- Forward --------
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        x:    [batch, seq_len, input_dim]
        mask: [batch, seq_len] (1=valid, 0=pad)
        """
        # 1) 임베딩
        embedded = self.keypoint_embedding(x)  # [B, T, H]

        # 2) 위치 인코딩(옵션)
        if self.use_positional_encoding:
            embedded = embedded.transpose(0, 1)      # [T, B, H]
            embedded = self.pos_encoding(embedded)
            embedded = embedded.transpose(0, 1)      # [B, T, H]

        # 3) BiLSTM 스택
        out = embedded
        for i, lstm in enumerate(self.lstm_layers):
            if mask is not None:
                lengths = mask.sum(dim=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(out, lengths, batch_first=True, enforce_sorted=False)
                packed_out, _ = lstm(packed)
                out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            else:
                out, _ = lstm(out)

            # (선택) 동일 차원일 때 잔차 연결
            if i > 0 and out.size(-1) == embedded.size(-1):
                out = out + embedded

        # 4) 어텐션(옵션)
        if self.use_attention:
            out_T = out.transpose(0, 1)                # [T, B, 2H]
            out_T = self.attention(out_T, mask)        # [T, B, 2H]
            out = out_T.transpose(0, 1)                # [B, T, 2H]

        # 5) 마스킹 평균 풀링
        if mask is not None:
            m = mask.unsqueeze(-1).float()             # [B, T, 1]
            summed = (out * m).sum(dim=1)              # [B, 2H]
            lengths = mask.sum(dim=1, keepdim=True).float()
            pooled = summed / (lengths + 1e-8)
        else:
            pooled = out.mean(dim=1)                   # [B, 2H]

        # 6) 분류
        logits = self.classifier(pooled)               # [B, C]

        outputs = {'logits': logits}
        if return_features:
            outputs['features'] = pooled
        return outputs

    # -------- Helper (inference) --------
    def predict(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            probs = F.softmax(self.forward(x, mask)['logits'], dim=-1)
        return probs

    def get_predictions(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = self.predict(x, mask)
        preds = torch.argmax(probs, dim=-1)
        maxp = torch.max(probs, dim=-1)[0]
        return preds, maxp


# =========================
# Model Wrapper
# =========================
class SignLanguageModel(nn.Module):
    """배치 dict 입출력 래퍼(학습 루프와 맞물림)"""
    def __init__(self, vocab_size: int, **model_kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.model = BiLSTMSignClassifier(num_classes=vocab_size, **model_kwargs)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        seqs = batch['sequence']                  # [B, T, D]
        masks = batch.get('sequence_mask')        # [B, T]
        out = self.model(seqs, masks)
        if 'label' in batch:
            labels = batch['label']               # [B]
            out['loss'] = F.cross_entropy(out['logits'], labels)
        return out

    def predict_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            seqs = batch['sequence']
            masks = batch.get('sequence_mask')
            out = self.model(seqs, masks, return_features=True)
            probs = F.softmax(out['logits'], dim=-1)
            preds = torch.argmax(probs, dim=-1)
            maxp = torch.max(probs, dim=-1)[0]
            return {
                'predictions': preds,
                'probabilities': probs,
                'max_probabilities': maxp,
                'features': out.get('features')
            }


# =========================
# Factory
# =========================
def create_model(vocab_size: int, device: Optional[torch.device] = None, **kwargs) -> SignLanguageModel:
    model = SignLanguageModel(vocab_size=vocab_size, **kwargs)
    if device is not None:
        model = model.to(device)
    return model


# =========================
# Quick self test
# =========================
def main():
    print("🧪 수어 인식 모델 테스트")
    vocab_size = 1504  # 예: 1500 단어 + 특수토큰
    model = create_model(vocab_size=vocab_size)
    print(f"모델 생성 완료: {count_parameters(model):,} 파라미터")

    # 더미 배치
    B, T, D = 4, 120, TOTAL_LANDMARKS * 3
    batch = {
        'sequence': torch.randn(B, T, D),
        'sequence_mask': torch.ones(B, T),
        'label': torch.randint(0, vocab_size, (B,))
    }

    # 순전파
    out = model(batch)
    print(f"로짓 크기: {out['logits'].shape}, 손실: {out['loss'].item():.4f}")

    # 예측
    pred_out = model.predict_batch(batch)
    print(f"예측: {pred_out['predictions']}, 최대확률: {pred_out['max_probabilities']}")

    print("✅ 모델 테스트 완료")

if __name__ == "__main__":
    main()
