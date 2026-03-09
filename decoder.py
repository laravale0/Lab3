import numpy as np


def create_causal_mask(seq_len):
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    return mask


def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


np.random.seed(33)

seq_len = 5
d_k = 8

Q = np.random.laplace(loc=0.0, scale=1.0, size=(seq_len, d_k))
K = np.random.beta(a=0.4, b=3.0, size=(seq_len, d_k))

mask = create_causal_mask(seq_len)

raw_scores = (Q @ K.T) / np.sqrt(d_k)
masked_scores = raw_scores + mask
attention_weights = softmax(masked_scores)

print("Mascara causal:")
print(mask)
print("\nQ (Laplace, loc=0, escala=1):")
print(np.round(Q, 4))
print("\nK (Beta, a=0.4, b=3.0):")
print(np.round(K, 4))
print("\nPesos de atencao apos mascaramento:")
print(np.round(attention_weights, 4))
print("\nSoma de cada linha (deve ser 1.0):")
print(np.round(attention_weights.sum(axis=-1), 6))
