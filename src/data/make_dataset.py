import numpy as np


class DataGenerator:
    def __init__(self, path):
        self.path = path
        with open(path) as f:
            data = f.read().lower()
        self.chars = list(set(data))
        self.char_to_idx = {ch: i for (i, ch) in enumerate(self.chars)}
        self.idx_to_char = {i: ch for (i, ch) in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        with open(path) as f:
            self.examples = [x.lower().strip() for x in f.readlines()]

    def generate_example(self, idx):
        example_chars = self.examples[idx]
        # 文字列をインデックスに変換
        example_char_idx = [self.char_to_idx[ch] for ch in example_chars]
        # X: 入力配列、開始文字として改行文字（\n）のインデックスを追加し、その後に例の文字列のインデックスを追加
        X = [self.char_to_idx['\n']] + example_char_idx
        # Y: 出力配列、例の文字列のインデックスの後に終了文字として改行文字（\n）のインデックスを追加
        Y = example_char_idx + [self.char_to_idx['\n']]

        return np.array(X), np.array(Y)
