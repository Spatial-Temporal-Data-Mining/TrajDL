# Copyright 2024 All authors of TrajDL
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains accuracy (ACC), accuracy at K (ACC_K), and Macro F1 score in TUL tasks.

These three metrics are integrated into TULMetrics.
"""

from typing import Dict, List

import numpy as np


class TULMetrics:
    """Class for calculating accuracy metrics in TUL tasks."""

    def __init__(self, num_users: int, topk: int = 5):
        """
        Initialize TULMetrics with the number of users and the evaluation parameter top-k.

        Parameters
        ----------
        num_users : int
            The number of users in the TUL task. Must be greater than 1.
        topk : int, optional
            Evaluate on top-k predicted users (default is 5). Must be less than num_users.

        Raises
        ------
        ValueError
            If `num_users` is less than or equal to 1 or less than or equal to `topk`.
        """
        self.num_users = num_users
        self._k = topk
        if num_users <= 1:
            raise ValueError("`num_users` should be greater than 1.")
        if num_users <= topk:
            raise ValueError("`num_users` should be greater than `topk`.")
        self.num_samples = 0
        self.acc = 0
        self.acc_topk = 0

        # eval_dict[user_idx][0] tracks correct top-1 predictions.
        # eval_dict[user_idx][1] tracks the number of times any user index is predicted as the top-1.
        # eval_dict[user_idx][2] tracks how many times each user index appears in the actual labels.
        # eval_dict[user_idx][3] stores the calculated Macro-F1 score for each user index.
        self.eval_dict = self.init_state()

    @property
    def k(self) -> int:
        """Return the top-k evaluation value."""
        return self._k

    def init_state(self) -> Dict[int, List[float]]:
        """
        Initialize the state for evaluation metrics.

        Returns
        -------
        Dict[int, List[float]]
            A dictionary where keys are user indices and values are lists
            tracking prediction statistics for each user.
        """
        return {user_idx: [0, 0, 0, 0] for user_idx in range(self.num_users)}

    def reset(self):
        """Reset all evaluation metrics to their initial state."""
        self.num_samples = 0
        self.acc = 0
        self.acc_topk = 0
        self.eval_dict = self.init_state()

    def update(self, preds: np.ndarray, targets: np.ndarray):
        """
        Update the metrics based on the predictions and actual targets.

        Parameters
        ----------
        preds : np.ndarray
            Array of shape (B, num_users) containing predicted scores for each user.
        targets : np.ndarray
            Array of shape (B,) containing the actual user indices (targets).
        """
        for idx in range(targets.shape[0]):
            self.num_samples += 1
            vec = preds[idx]
            user_idx = targets[idx]
            self.eval_dict[user_idx][2] += 1
            topk = np.argpartition(a=-vec, kth=self.k)[: self.k]
            top1 = np.argpartition(a=-vec, kth=1)[0]
            self.eval_dict[top1][1] += 1
            for index in range(self.k):
                if topk[index] == user_idx:
                    self.acc_topk += 1
                    break
            if top1 == user_idx:
                self.acc += 1
                self.eval_dict[user_idx][0] += 1

    def value(self) -> Dict[str, float]:
        """Compute accuracy, accuracy at K, and Macro-F1 score.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the computed metrics:
                - 'acc': Accuracy
                - 'acc_topk': Accuracy at K
                - 'macro-f1': Macro F1 score
        """
        macro = 0
        for i in self.eval_dict.keys():
            if self.eval_dict[i][1] + self.eval_dict[i][2] > 0:
                self.eval_dict[i][3] = (2 * self.eval_dict[i][0]) / (
                    self.eval_dict[i][1] + self.eval_dict[i][2]
                )
                macro += self.eval_dict[i][3]
        macro = macro * 100 / len(self.eval_dict)
        acc = self.acc * 100 / self.num_samples
        acc_topk = self.acc_topk * 100 / self.num_samples
        print(
            "\nacc1: {:.2f}%, acck: {:.2f}%, macro-f1: {:.2f}%".format(
                acc, acc_topk, macro
            ),
            flush=True,
        )
        return {"acc": acc, "acc_topk": acc_topk, "macro-f1": macro}
