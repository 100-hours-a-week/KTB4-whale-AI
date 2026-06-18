from collections import defaultdict, Counter
import re

class BPETokenizer:
    def __init__(self, vocab_size: int):
        """
        vocab_size: 최종적으로 만들 어휘 사전의 크기
        """
        self.vocab_size = vocab_size
        self.vocab = {} # 최종 vocabulary
        self.merge_rules = [] # 병합 규칙 (학습된 순서대로 저장)
        self.token_to_id = {}
        self.id_to_token = {}

    def train(self, corpus: list[str]):
        """
        BPE 학습 메인 함수
        - corpus: 학습에 사용할 텍스트 리스트
        """

        # 1. 초기 vocab 구성
        vocab = {}
        for sentence in corpus:             
            for word in sentence.split(): # 입력된 corpus를 단어 단위로 구분
                # 각 단어를 문자 단위로 쪼갠 후 </w>를 붙여 초기 vocab 만들기
                word = ' '.join(list(word)) + ' </w>'
                vocab[word] = vocab.get(word, 0) + 1
                

        # 2. 반복적으로 병합
        # while 루프를 돌면서
        while len(vocab) < self.vocab_size: # 목표: vocab_size에 도달하거나
            # pair 빈도 계산(_get_stats())
            pairs = self._get_stats(vocab)
            if not pairs: # 목표: 더 이상 병합할 pair가 없을 때까지 반복
                break

            # 가장 빈번한 pair 선택
            best_pair = max(pairs, key=pairs.get)

            # _merge_vocab()로 병합
            vocab = self._merge_vocab(best_pair, vocab)

            # merge_rules에 기록
            self.merge_rules.append(best_pair)

        # 3. 최종 vocabulary 및 ID 매핑 생성
        self.vocab = vocab

        # 최종 vocab에서 고유 토큰들을 추출하여
        all_tokens = set()
        for word in vocab.keys():
            all_tokens.update(word.split())

        # token_to_id, id_to_token 생성(토큰을 정렬하여 일관된 순서로 ID 부여)
        sorted_tokens = sorted(all_tokens) # 오름차순

        self.token_to_id = {token: idx for idx, token in enumerate(sorted_tokens)}        
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        pass

    def _get_stats(self, vocab: dict) -> dict:
        """
        현재 vocab에서 인접한 pair 빈도 계산
        - return: {(token1, token2): 빈도수, ...}
        """
        pairs = defaultdict(int)

        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pair = (symbols[i], symbols[i+1])
                pairs[pair] += freq
        
        return pairs

    def _merge_vocab(self, pair: tuple, vocab: dict) -> dict:
        """
        가장 빈번한 pair를 병합
        
        args:
            pair: 병합할 두 토큰 (예시: ('l', 'o'))
            vocab: 현재 단어-빈도 딕셔너리

        returns:
            병합이 적용된 새로운 vocab 딕셔너리
        """
        new_vocab = {}

        # pair를 문자열로 변환
        # 예시: ('l', 'o') -> 'l o'(검색용), 'lo' (대체용)
        bigram = ' '.join(pair) # 검색용
        replacement = ''.join(pair) # 대체용

        for word, freq in vocab.items():
            # 단어 내에서 bigram을 replacement로 치환
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq

        return new_vocab

    def encode(self, text: str) -> list[int]:
        """텍스트를 token ID로 변환"""
        pass

    def decode(self, token_ids: list[int]) -> str:
        """token ID를 텍스트로 복원"""
        pass