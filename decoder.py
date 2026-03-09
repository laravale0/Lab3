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


d_model = 512
d_k_cross = 64

encoder_out = np.random.randn(1, 10, d_model)
decoder_state = np.random.randn(1, 4, d_model)

W_Q = np.random.randn(d_model, d_k_cross)
W_K = np.random.randn(d_model, d_k_cross)
W_V = np.random.randn(d_model, d_k_cross)


def cross_attention(encoder_out, decoder_state):
    Q = decoder_state @ W_Q
    K = encoder_out @ W_K
    V = encoder_out @ W_V

    scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_k_cross)
    weights = softmax(scores)
    output = weights @ V

    return output, weights


cross_out, cross_weights = cross_attention(encoder_out, decoder_state)

print("\n--- Cross-Attention ---")
print(f"encoder_out: {encoder_out.shape}")
print(f"decoder_state: {decoder_state.shape}")
print(f"\nForma da saida: {cross_out.shape}")
print(f"\nPesos de atencao (decoder -> encoder):")
print(np.round(cross_weights[0], 4))
print(f"\nSoma de cada linha (deve ser 1.0):")
print(np.round(cross_weights[0].sum(axis=-1), 6))


vocab_size = 10000
EOS_IDX = 9999

vocab = {
    0: "<START>",
    EOS_IDX: "<EOS>",
}


def generate_next_token(current_sequence, encoder_out):
    step = len(current_sequence)
    context_bias = encoder_out[0].mean()
    logits = np.random.randn(vocab_size) + float(context_bias) * 0.01
    logits[EOS_IDX] += step * 0.9
    probs = softmax(logits)
    return probs


sequence = ["<START>"]

print("\n--- Loop de Inferencia Auto-Regressivo ---")

while True:
    probs = generate_next_token(sequence, encoder_out)
    next_idx = int(np.argmax(probs))
    next_token = vocab.get(next_idx, f"tok_{next_idx}")
    sequence.append(next_token)

    print(f"Passo {len(sequence) - 1}: '{next_token}' (idx={next_idx}, p={probs[next_idx]:.4f})")

    if next_token == "<EOS>":
        break

print(f"\nFrase gerada: {' '.join(sequence)}")
