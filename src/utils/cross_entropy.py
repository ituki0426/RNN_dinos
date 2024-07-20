import numpy as np
def coross_entropy(y_preds, targets, len_vocab):
    """
        Computes the cross-entropy loss for a given sequence of predicted probabilities and true targets.

        Parameters:
            y_preds (ndarray): Array of shape (sequence_length, vocab_size) containing the predicted probabilities for each time step.
            targets (ndarray): Array of shape (sequence_length, 1) containing the true targets for each time step.

        Returns:
            float: Cross-entropy loss.
        """
    # calculate cross-entropy loss
    return sum(-np.log(y_preds[t][targets[t], 0]) for t in range(len_vocab))
