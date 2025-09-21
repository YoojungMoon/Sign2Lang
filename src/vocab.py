#!/usr/bin/env python3
"""
SU:DA - 수어 단어 사전 관리 모듈
수어 단어와 인덱스 간의 매핑을 관리
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from utils import setup_logger, load_json, save_json, DATA_PROCESSED_PATH

class SignVocabulary:
    """수어 단어 사전 클래스"""
    
    def __init__(self, vocab_path: Optional[Path] = None):
        """
        Args:
            vocab_path: 사전 파일 경로 (None이면 기본 경로 사용)
        """
        self.logger = setup_logger("SignVocabulary")
        
        # 기본 사전 파일 경로
        if vocab_path is None:
            vocab_path = DATA_PROCESSED_PATH / "vocab" / "vocab.json"
        
        self.vocab_path = vocab_path
        
        # 특수 토큰 정의
        self.special_tokens = {
            "PAD": "<PAD>",      # 패딩 토큰
            "UNK": "<UNK>",      # 알 수 없는 단어
            "SOS": "<SOS>",      # 시작 토큰 (Start of Sequence)
            "EOS": "<EOS>"       # 종료 토큰 (End of Sequence)
        }
        
        # 사전 초기화
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.vocab_size: int = 0
        
        # 사전 로딩
        self.load_vocabulary()
    
    def load_vocabulary(self) -> bool:
        """사전 파일에서 단어 사전 로딩"""
        if not self.vocab_path.exists():
            self.logger.warning(f"사전 파일이 존재하지 않습니다: {self.vocab_path}")
            self.logger.info("빈 사전으로 초기화합니다.")
            self._initialize_empty_vocab()
            return False
        
        try:
            vocab_data = load_json(self.vocab_path)
            
            # word -> idx 매핑
            self.word_to_idx = vocab_data
            
            # idx -> word 매핑 (역방향)
            self.idx_to_word = {idx: word for word, idx in vocab_data.items()}
            
            # 사전 크기
            self.vocab_size = len(vocab_data)
            
            self.logger.info(f"사전 로딩 완료: {self.vocab_size}개 단어")
            self.logger.info(f"특수 토큰: {list(self.special_tokens.values())}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"사전 로딩 실패: {e}")
            self._initialize_empty_vocab()
            return False
    
    def _initialize_empty_vocab(self):
        """빈 사전으로 초기화 (특수 토큰만)"""
        self.word_to_idx = {
            self.special_tokens["PAD"]: 0,
            self.special_tokens["UNK"]: 1,
            self.special_tokens["SOS"]: 2,
            self.special_tokens["EOS"]: 3
        }
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
    
    def save_vocabulary(self, save_path: Optional[Path] = None):
        """사전을 파일로 저장"""
        if save_path is None:
            save_path = self.vocab_path
        
        try:
            # 디렉토리 생성
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # JSON 형태로 저장
            save_json(self.word_to_idx, save_path)
            
            # 텍스트 형태로도 저장
            txt_path = save_path.with_suffix('.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                for word, idx in sorted(self.word_to_idx.items(), key=lambda x: x[1]):
                    f.write(f"{idx}\t{word}\n")
            
            self.logger.info(f"사전 저장 완료: {save_path}")
            
        except Exception as e:
            self.logger.error(f"사전 저장 실패: {e}")
    
    def add_word(self, word: str) -> int:
        """새 단어를 사전에 추가"""
        if word in self.word_to_idx:
            return self.word_to_idx[word]
        
        # 새 인덱스 할당
        new_idx = self.vocab_size
        self.word_to_idx[word] = new_idx
        self.idx_to_word[new_idx] = word
        self.vocab_size += 1
        
        self.logger.debug(f"새 단어 추가: '{word}' -> {new_idx}")
        
        return new_idx
    
    def add_words(self, words: List[str]) -> List[int]:
        """여러 단어를 한번에 추가"""
        indices = []
        for word in words:
            idx = self.add_word(word)
            indices.append(idx)
        
        self.logger.info(f"{len(words)}개 단어 추가 완료")
        return indices
    
    def word_to_index(self, word: str) -> int:
        """단어를 인덱스로 변환"""
        return self.word_to_idx.get(word, self.word_to_idx[self.special_tokens["UNK"]])
    
    def index_to_word(self, index: int) -> str:
        """인덱스를 단어로 변환"""
        return self.idx_to_word.get(index, self.special_tokens["UNK"])
    
    def words_to_indices(self, words: List[str]) -> List[int]:
        """단어 리스트를 인덱스 리스트로 변환"""
        return [self.word_to_index(word) for word in words]
    
    def indices_to_words(self, indices: List[int]) -> List[str]:
        """인덱스 리스트를 단어 리스트로 변환"""
        return [self.index_to_word(idx) for idx in indices]
    
    def encode(self, text: Union[str, List[str]], 
               add_special_tokens: bool = False) -> List[int]:
        """
        텍스트를 인덱스 시퀀스로 인코딩
        
        Args:
            text: 단일 단어(str) 또는 단어 리스트(List[str])
            add_special_tokens: SOS, EOS 토큰 추가 여부
        
        Returns:
            인덱스 리스트
        """
        if isinstance(text, str):
            words = [text]
        else:
            words = text
        
        indices = self.words_to_indices(words)
        
        if add_special_tokens:
            sos_idx = self.word_to_idx[self.special_tokens["SOS"]]
            eos_idx = self.word_to_idx[self.special_tokens["EOS"]]
            indices = [sos_idx] + indices + [eos_idx]
        
        return indices
    
    def decode(self, indices: List[int], 
               remove_special_tokens: bool = True) -> List[str]:
        """
        인덱스 시퀀스를 텍스트로 디코딩
        
        Args:
            indices: 인덱스 리스트
            remove_special_tokens: 특수 토큰 제거 여부
        
        Returns:
            단어 리스트
        """
        words = self.indices_to_words(indices)
        
        if remove_special_tokens:
            special_token_values = set(self.special_tokens.values())
            words = [word for word in words if word not in special_token_values]
        
        return words
    
    def contains_word(self, word: str) -> bool:
        """단어가 사전에 있는지 확인"""
        return word in self.word_to_idx
    
    def contains_index(self, index: int) -> bool:
        """인덱스가 유효한지 확인"""
        return index in self.idx_to_word
    
    def get_vocab_size(self) -> int:
        """사전 크기 반환"""
        return self.vocab_size
    
    def get_special_token_idx(self, token_name: str) -> int:
        """특수 토큰의 인덱스 반환"""
        if token_name not in self.special_tokens:
            raise ValueError(f"알 수 없는 특수 토큰: {token_name}")
        
        token = self.special_tokens[token_name]
        return self.word_to_idx[token]
    
    def get_pad_idx(self) -> int:
        """PAD 토큰 인덱스 반환"""
        return self.get_special_token_idx("PAD")
    
    def get_unk_idx(self) -> int:
        """UNK 토큰 인덱스 반환"""
        return self.get_special_token_idx("UNK")
    
    def get_sos_idx(self) -> int:
        """SOS 토큰 인덱스 반환"""
        return self.get_special_token_idx("SOS")
    
    def get_eos_idx(self) -> int:
        """EOS 토큰 인덱스 반환"""
        return self.get_special_token_idx("EOS")
    
    def get_word_list(self, include_special_tokens: bool = True) -> List[str]:
        """모든 단어 리스트 반환"""
        if include_special_tokens:
            return list(self.word_to_idx.keys())
        else:
            special_token_values = set(self.special_tokens.values())
            return [word for word in self.word_to_idx.keys() 
                    if word not in special_token_values]
    
    def get_word_frequencies(self, word_list: List[str]) -> Dict[str, int]:
        """단어 빈도 계산"""
        freq = {}
        for word in word_list:
            freq[word] = freq.get(word, 0) + 1
        return freq
    
    def filter_by_frequency(self, word_list: List[str], 
                          min_freq: int = 1) -> List[str]:
        """빈도가 임계값 이상인 단어들만 필터링"""
        frequencies = self.get_word_frequencies(word_list)
        return [word for word, freq in frequencies.items() if freq >= min_freq]
    
    def print_vocabulary_info(self):
        """사전 정보 출력"""
        print("=" * 50)
        print("📚 수어 단어 사전 정보")
        print("=" * 50)
        print(f"📊 총 단어 수: {self.vocab_size}")
        print(f"🔤 특수 토큰 수: {len(self.special_tokens)}")
        print(f"💬 일반 단어 수: {self.vocab_size - len(self.special_tokens)}")
        print()
        print("🏷️ 특수 토큰:")
        for name, token in self.special_tokens.items():
            idx = self.word_to_idx[token]
            print(f"  {name}: '{token}' -> {idx}")
        print()
        
        # 일반 단어 몇 개 예시
        regular_words = self.get_word_list(include_special_tokens=False)
        if regular_words:
            print("📝 수어 단어 예시 (처음 10개):")
            for i, word in enumerate(regular_words[:10]):
                idx = self.word_to_idx[word]
                print(f"  {word} -> {idx}")
            if len(regular_words) > 10:
                print(f"  ... 외 {len(regular_words) - 10}개")
        print("=" * 50)
    
    def __len__(self) -> int:
        """사전 크기 반환 (len() 함수용)"""
        return self.vocab_size
    
    def __contains__(self, item) -> bool:
        """단어/인덱스 포함 여부 확인 (in 연산자용)"""
        if isinstance(item, str):
            return self.contains_word(item)
        elif isinstance(item, int):
            return self.contains_index(item)
        else:
            return False
    
    def __getitem__(self, key):
        """인덱싱 접근 (vocab[word] 또는 vocab[index])"""
        if isinstance(key, str):
            return self.word_to_index(key)
        elif isinstance(key, int):
            return self.index_to_word(key)
        else:
            raise TypeError("키는 문자열(단어) 또는 정수(인덱스)여야 합니다")
    
    def __repr__(self) -> str:
        return f"SignVocabulary(size={self.vocab_size}, path='{self.vocab_path}')"

def create_vocabulary_from_words(words: List[str], 
                                save_path: Optional[Path] = None) -> SignVocabulary:
    """
    단어 리스트로부터 새로운 사전 생성
    
    Args:
        words: 수어 단어 리스트
        save_path: 저장할 경로 (None이면 기본 경로)
    
    Returns:
        생성된 SignVocabulary 객체
    """
    # 빈 사전 생성
    vocab = SignVocabulary(vocab_path=save_path)
    
    # 단어들 추가
    unique_words = sorted(set(words))  # 중복 제거 및 정렬
    vocab.add_words(unique_words)
    
    # 저장
    if save_path:
        vocab.save_vocabulary(save_path)
    
    return vocab

def main():
    """테스트용 메인 함수"""
    # 기본 사전 로딩 테스트
    vocab = SignVocabulary()
    
    # 사전 정보 출력
    vocab.print_vocabulary_info()
    
    # 기본 사용법 테스트
    if vocab.vocab_size > 4:  # 특수 토큰 외에 일반 단어가 있는 경우
        print("\n🧪 사용법 테스트:")
        
        # 첫 번째 일반 단어로 테스트
        regular_words = vocab.get_word_list(include_special_tokens=False)
        if regular_words:
            test_word = regular_words[0]
            
            # 인코딩/디코딩 테스트
            encoded = vocab.encode(test_word)
            decoded = vocab.decode(encoded)
            
            print(f"단어: '{test_word}'")
            print(f"인코딩: {encoded}")
            print(f"디코딩: {decoded}")
            
            # 특수 토큰 포함 테스트
            encoded_with_special = vocab.encode(test_word, add_special_tokens=True)
            decoded_with_special = vocab.decode(encoded_with_special, remove_special_tokens=False)
            
            print(f"특수토큰 포함 인코딩: {encoded_with_special}")
            print(f"특수토큰 포함 디코딩: {decoded_with_special}")

if __name__ == "__main__":
    main()