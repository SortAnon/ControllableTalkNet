""" Copyright 2021 NVIDIA Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License."""

import numpy as np


def forward_extractor(tokens, log_probs, blank):
    """Computes states f and p."""
    n, m = len(tokens), log_probs.shape[0]
    # `f[s, t]` -- max sum of log probs for `s` first codes
    # with `t` first timesteps with ending in `tokens[s]`.
    f = np.empty((n + 1, m + 1), dtype=float)
    f.fill(-(10 ** 9))
    p = np.empty((n + 1, m + 1), dtype=int)
    f[0, 0] = 0.0  # Start
    for s in range(1, n + 1):
        c = tokens[s - 1]
        for t in range((s + 1) // 2, m + 1):
            f[s, t] = log_probs[t - 1, c]
            # Option #1: prev char is equal to current one.
            if s == 1 or c == blank or c == tokens[s - 3]:
                options = f[s : (s - 2 if s > 1 else None) : -1, t - 1]
            else:  # Is not equal to current one.
                options = f[s : (s - 3 if s > 2 else None) : -1, t - 1]
            f[s, t] += np.max(options)
            p[s, t] = np.argmax(options)
    return f, p


def backward_extractor(f, p):
    """Computes durs from f and p."""
    n, m = f.shape
    n -= 1
    m -= 1
    durs = np.zeros(n, dtype=int)
    if f[-1, -1] >= f[-2, -1]:
        s, t = n, m
    else:
        s, t = n - 1, m
    while s > 0:
        durs[s - 1] += 1
        s -= p[s, t]
        t -= 1
    assert durs.shape[0] == n
    assert np.sum(durs) == m
    assert np.all(durs[1::2] > 0)
    return durs
