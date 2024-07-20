import numpy as np
import os
from utils.adam import adam
from utils.softmax import softmax
from data.make_dataset import DataGenerator

class RNN:
    def __init__(self, hidden_size, data_generator, sequence_length, learning_rate):
        """
        Initializes an instance of the RNN class.

        Parameters
        ----------
        hidden_size : int
            The number of hidden units in the RNN.
        vocab_size : int
            The size of the vocabulary used by the RNN.
        sequence_length : int
            The length of the input sequences fed to the RNN.
        learning_rate : float
            The learning rate used during training.
        """
        self.hidden_size = hidden_size
        self.data_generator = data_generator
        self.vocab_size = data_generator.vocab_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.X = None

        self.param = {
            # np.random.uniform(a, b, size) は、範囲 [a, b) で一様分布の乱数を生成
            # -np.sqrt(1. / self.vocab_size) から np.sqrt(1. / self.vocab_size) の範囲で乱数を生成します。
            "Wax": np.random.uniform(-np.sqrt(1. / self.vocab_size), np.sqrt(1. / self.vocab_size), (self.hidden_size, self.vocab_size)),
            "Waa": np.random.uniform(-np.sqrt(1. / self.hidden_size), np.sqrt(1. / self.hidden_size), (self.hidden_size, self.hidden_size)),
            "Wya": np.random.uniform(-np.sqrt(1. / self.hidden_size), np.sqrt(1. / self.hidden_size), (self.vocab_size, self.hidden_size)),
            "ba": np.zeros((self.hidden_size, 1)),
            "by": np.zeros((self.vocab_size, 1))
        }
        self.grad = {
            "dWax": np.zeros_like(self.param["Wax"]),
            "dWaa": np.zeros_like(self.param["Waa"]),
            "dWya": np.zeros_like(self.param["Wya"]),
            "dba": np.zeros_like(self.param["ba"]),
            "dby": np.zeros_like(self.param["by"])
        }
        self.grad_adam = {
            "mWax": np.zeros_like(self.param["Wax"]),
            "vWax": np.zeros_like(self.param["Wax"]),
            "mWaa": np.zeros_like(self.param["Waa"]),
            "vWaa": np.zeros_like(self.param["Waa"]),
            "mWya": np.zeros_like(self.param["Wya"]),
            "vWya": np.zeros_like(self.param["Wya"]),
            "mba": np.zeros_like(self.param["ba"]),
            "vba": np.zeros_like(self.param["ba"]),
            "mby": np.zeros_like(self.param["by"]),
            "vby": np.zeros_like(self.param["by"])
        }
        

    def forward(self, X, a_prev):
        """

        Parameters::
            - X(ndarray): (seq_lengthm vocab_size)形状が入力データのリスト
            - a_prev(ndarray): 形状が（hidden_size,1）前の時刻の隠れ状態
        Returns:
            - x(dict): 形状が (seq_length, vocab_size, 1) の入力データを格納する辞書。キーは 0 から seq_length-1 まで
            - a (dict): 各時刻の隠れ活性化を格納する辞書。キーは 0 から seq_length-1 まで
            - y_pred (dict): 各時刻の出力確率を格納する辞書。キーは 0 から seq_length-1 まで
        """
        x, a, y_pred = {}, {}, {}
        self.X = X

        # 初期隠れ状態として、前の時刻の隠れ状態' a_prev 'を使用
        a[-1] = np.copy(a_prev)

        for t in range(len(self.X)):
            x[t] = np.zeros((self.vocab_size, 1))
            if (self.X[t] != None):
                # 入力データがNoneでない場合、ワンホットエンコーディングを適用する
                # self.X[t]]は現在の入力文字のインデックスであり、そのインデックスに対応する位置に1を設定
                # keyがtのxは、形状が (vocab_size, 1) のワンホットエンコーディングされた入力データ
                x[t][self.X[t]] = 1
                a[t] = np.tanh(np.dot(self.param["Wax"], x[t]) +
                               np.dot(self.param["Waa"], a[t-1]) + self.param["ba"])
                y_pred[t] = softmax(np.dot(self.param["Wya"], a[t]) + self.param["by"])
        return x, a, y_pred

    def backward(self, x, a, y_preds, targets):
        da_next = np.zeros_like(a[0])

        for t in reversed(range(len(self.X))):

            dy_preds = np.copy(y_preds[t])
            # softmaxの逆伝播
            dy_preds[targets[t]] -= 1

            da = np.dot(self.param["Waa"].T, da_next) + np.dot(self.param["Wya"].T, dy_preds)
            dtanh = (1 - np.power(a[t], 2))
            da_unactivated = dtanh * da

            self.grad["dba"] += da_unactivated
            self.grad["dWax"] += np.dot(da_unactivated, x[t].T)
            self.grad["dWaa"] += np.dot(da_unactivated, a[t-1].T)

            da_next = da_unactivated

            self.grad["dWya"] += np.dot(dy_preds, a[t].T)

            for grad in [self.grad["dWax"], self.grad["dWaa"], self.grad["dWya"], self.grad["dba"], self.grad["dby"]]:
                np.clip(grad, -1, 1, out=grad)

    def loss(self, y_preds, targets):
        # calculate the loss
        return sum(-np.log(y_preds[t][targets[t], 0]) for t in range(len(self.X)))

    def sample(self):
        """
        Sample a sequence of characters from the RNN.

        Args:
            None
        Returns:
            list : A list of integers representing the sampled characters.  
        """
        x = np.zeros((self.vocab_size, 1))
        a_prev = np.zeros((self.hidden_size, 1))

        indices = []
        idx = -1

        counter = 0
        max_chars = 50
        newline_character = self.data_generator.char_to_idx['\n']

        while (idx != newline_character and counter != max_chars):
            # compute the hidden state
            a = np.tanh(np.dot(self.param["Wax"], x) +
                        np.dot(self.param["Waa"], a_prev) + self.param["ba"])
            # compute the output probabilities
            y = softmax(np.dot(self.param["Wya"], a) + self.param["by"])

            idx = np.random.choice(range(self.vocab_size), p=y.ravel())

            # set the input for the next time step
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1

            # store the sampled character index in the list
            indices.append(idx)

            # update the previous hidden state
            a_prev = a
            counter += 1

        return indices

    def train(self, generated_names=5):
        """
        Train the RNN on a dataset using backpropagation through time (BPTT).

        Args:
        - generated_names: an integer indicating how many example names to generate during training.

        Returns:
        - None
        """
        iter_num = 0
        threshold = 5
        # モデルが何も学習していない場合の理論的な損失値の上限
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.sequence_length
        while (smooth_loss > threshold):
            a_prev = np.zeros((self.hidden_size, 1))
            idx = iter_num % self.vocab_size
            inputs, targets = self.data_generator.generate_example(idx)

            # Forward pass
            x, a, y_preds = self.forward(inputs, a_prev)

            # Back propagation
            self.backward(x, a, y_preds, targets)

            # caluculate and update loss
            loss = self.loss(y_preds, targets)
            adam(self.param, self.grad_adam, self.grad, self.learning_rate, beta1= 0.9, beta2 = 0.999, epsilon = 1e-8, L2_reg = 1e-4)
            smooth_loss = 0.999 * smooth_loss + 0.001 * loss

            a_prev = a[len(self.X) - 1]
            if iter_num % 500 == 0:
                print("Iteration: {}, Loss: {}".format(iter_num, smooth_loss))
                for i in range(generated_names):
                    sample_idx = self.sample()
                    txt = ''.join(
                        self.data_generator.idx_to_char[idx] for idx in sample_idx)
                    txt = txt.title()  # capitalize first character
                    print('%s' % (txt, ), end='')
            iter_num += 1

    def predict(self, start):
        """
        Generate a sequence of characters using the trained self, starting from the given start sequence.
        The generated sequence may contain a maximum of 50 characters or a newline character.

        Args:
        - start: a string containing the start sequence

        Returns:
        - txt: a string containing the generated sequence
        """
        x = np.zeros((self.vocab_size, 1))
        a_prev = np.zeros((self.hidden_size, 1))
        # Convert start sequence to indices
        chars = [ch for ch in start]
        indexs = []
        # char_to_index = {'\n': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26}
        # index_to_char = {0: '\n', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}
        for i in range(len(chars)):
            idx = self.data_generator.char_to_index[chars[i]]
            x[idx] = 1
            indexs.append(idx)
        # Generate characters
        max_chars = 50
        # the newline character
        newline_character = self.data_generator.char_to_idx['\n']
        counter = 0
        while (idx != newline_character and counter != max_chars):
            a = np.tanh(np.dot(self.param["Wax"], x) +
                        np.dot(self.param["Waa"], a_prev) + self.param["ba"])
            y_pred = softmax(np.dot(self.param["Wya"], a) + self.param["by"])
            idx = np.random.choice(range(self.vocab_size), p=y_pred.ravel())

            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            a_prev = a
            indexs.append(idx)
            counter += 1
        txt = ''.join(self.data_generator.index_to_char[i] for i in indexs)
        if txt[-1] == '\n':
            txt = txt[:-1]

        return txt


if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(__file__), '../data/dinos.txt')
    data_generator = DataGenerator(file_path)
    rnn = RNN(hidden_size=200, data_generator=data_generator,
              sequence_length=25, learning_rate=1e-3)
    rnn.train()
