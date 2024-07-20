import numpy as np
import os


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
        # np.random.uniform(a, b, size) は、範囲 [a, b) で一様分布の乱数を生成
        # -np.sqrt(1. / self.vocab_size) から np.sqrt(1. / self.vocab_size) の範囲で乱数を生成します。
        self.Wax = np.random.uniform(-np.sqrt(1. / self.vocab_size), np.sqrt(
            1. / self.vocab_size), (self.hidden_size, self.vocab_size))
        self.Waa = np.random.uniform(-np.sqrt(1. / self.hidden_size), np.sqrt(
            1. / self.hidden_size), (self.hidden_size, self.hidden_size))
        self.Wya = np.random.uniform(-np.sqrt(1. / self.hidden_size), np.sqrt(
            1. / self.hidden_size), (self.vocab_size, self.hidden_size))

        self.ba = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((self.vocab_size, 1))

        self.dWax, self.dWaa, self.dWya = np.zeros_like(
            self.Wax), np.zeros_like(self.Waa), np.zeros_like(self.Wya)
        self.dba, self.dby = np.zeros_like(self.ba), np.zeros_like(self.by)
        self.mWax = np.zeros_like(self.Wax)
        self.vWax = np.zeros_like(self.Wax)
        self.mWaa = np.zeros_like(self.Waa)
        self.vWaa = np.zeros_like(self.Waa)
        self.mWya = np.zeros_like(self.Wya)
        self.vWya = np.zeros_like(self.Wya)
        self.mba = np.zeros_like(self.ba)
        self.vba = np.zeros_like(self.ba)
        self.mby = np.zeros_like(self.by)
        self.vby = np.zeros_like(self.by)

    def softmax(self, x):
        """
        Coumputes the softmax activation for a given input array.
        Parameters:
            s(ndarray) : Input array.
        Returns:
            ndarray : Array of the same shape as 'x', containing the softmax activation values.
        """
        x = x - np.max(x)
        p = np.exp(x)
        return p / np.sum(p)

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
                a[t] = np.tanh(np.dot(self.Wax, x[t]) +
                               np.dot(self.Waa, a[t-1]) + self.ba)
                y_pred[t] = self.softmax(np.dot(self.Wya, a[t]) + self.by)
        return x, a, y_pred

    def backward(self, x, a, y_preds, targets):
        da_next = np.zeros_like(a[0])

        for t in reversed(range(len(self.X))):

            dy_preds = np.copy(y_preds[t])
            # softmaxの逆伝播
            dy_preds[targets[t]] -= 1

            da = np.dot(self.Waa.T, da_next) + np.dot(self.Wya.T, dy_preds)
            dtanh = (1 - np.power(a[t], 2))
            da_unactivated = dtanh * da

            self.dba += da_unactivated
            self.dWax += np.dot(da_unactivated, x[t].T)
            self.dWaa += np.dot(da_unactivated, a[t-1].T)

            da_next = da_unactivated

            self.dWya += np.dot(dy_preds, a[t].T)

            for grad in [self.dWax, self.dWaa, self.dWya, self.dba, self.dby]:
                np.clip(grad, -1, 1, out=grad)

    def loss(self, y_preds, targets):
        # calculate the loss
        return sum(-np.log(y_preds[t][targets[t], 0]) for t in range(len(self.X)))

    def adamw(self, beta1=0.9, beta2=0.999, epsilon=1e-8, L2_reg=1e-4):
        """
        Updates the RNN's parameters using the AdamW optimization algorithm.
        """
        self.mWax = beta1 * self.mWax + (1 - beta1) * self.dWax
        self.vWax = beta2 * self.vWax + (1 - beta2) * np.square(self.dWax)
        m_hat = self.mWax / (1 - beta1)
        v_hat = self.vWax / (1 - beta2)
        self.Wax -= self.learning_rate * \
            (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.Wax)

        self.mWaa = beta1 * self.mWaa + (1 - beta1) * self.dWaa
        self.vWaa = beta2 * self.vWaa + (1 - beta2) * np.square(self.dWaa)
        m_hat = self.mWaa / (1 - beta1)
        v_hat = self.vWaa / (1 - beta2)
        self.Waa -= self.learning_rate * \
            (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.Waa)

        self.mWya = beta1 * self.mWya + (1 - beta1) * self.dWya
        self.vWya = beta2 * self.vWya + (1 - beta2) * np.square(self.dWya)
        m_hat = self.mWya / (1 - beta1)
        v_hat = self.vWya / (1 - beta2)
        self.Wya -= self.learning_rate * \
            (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.Wya)

        self.mba = beta1 * self.mba + (1 - beta1) * self.dba
        self.vba = beta2 * self.vba + (1 - beta2) * np.square(self.dba)
        m_hat = self.mba / (1 - beta1)
        v_hat = self.vba / (1 - beta2)
        self.ba -= self.learning_rate * \
            (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.ba)

        self.mby = beta1 * self.mby + (1 - beta1) * self.dby
        self.vby = beta2 * self.vby + (1 - beta2) * np.square(self.dby)
        m_hat = self.mby / (1 - beta1)
        v_hat = self.vby / (1 - beta2)
        self.by -= self.learning_rate * \
            (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.by)

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
            a = np.tanh(np.dot(self.Wax, x) +
                        np.dot(self.Waa, a_prev) + self.ba)
            # compute the output probabilities
            y = self.softmax(np.dot(self.Wya, a) + self.by)

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
            self.adamw()
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
            a = np.tanh(np.dot(self.Wax, x) +
                        np.dot(self.Waa, a_prev) + self.ba)
            y_pred = self.softmax(np.dot(self.Wya, a) + self.by)
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
