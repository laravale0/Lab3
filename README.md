# Laboratório 3 - Implementando o Decoder

Implementação dos blocos matemáticos centrais do Decoder de um Transformer, cobrindo mascaramento causal, cross-attention e geração auto-regressiva.

## Estrutura do projeto

| Arquivo | Descrição |
|---|---|
| `decoder.py` | Mascara causal (Look-Ahead Mask) e Cross-Attention |

## Máscara Causal (Look-Ahead Mask)

Durante o treinamento, a frase inteira entra no Decoder ao mesmo tempo. A máscara causal impede que a posição `i` acesse posições futuras `i+1, i+2, ...`, inserindo `-inf` nos valores correspondentes antes do Softmax.

```
Attention(Q, K, V) = softmax( QK^T / sqrt(dk) + M ) * V
```

A função `create_causal_mask(seq_len)` retorna uma matriz `[seq_len, seq_len]` onde:
- Triangulo inferior + diagonal principal: `0`
- Triangulo superior: `-inf`

As matrizes Q e K foram geradas com distribuições intencionalmente distintas (seed=33):
- **Q**: distribuição de Laplace (loc=0, escala=1) — simétrica com pico acentuado em 0 e caudas pesadas
- **K**: distribuição Beta (a=0.4, b=3.0) — limitada em [0,1], fortemente assimétrica próxima de 0

Apos o Softmax, as posições mascaradas resultam em probabilidade `0.0`, confirmando que o modelo não acessa tokens futuros.

## Cross-Attention (Ponte Encoder-Decoder)

No Cross-Attention, o Decoder usa seu estado atual para consultar a saída do Encoder. As projeções vêm de fontes distintas: Q deriva do Decoder, enquanto K e V derivam do Encoder.

```
encoder_output : [1, 10, 512]   (frase francesa - 10 tokens)
decoder_state  : [1,  4, 512]   (traducao parcial - 4 tokens)
```

A função `cross_attention(encoder_out, decoder_state)` projeta cada tensor com matrizes de pesos `W_Q`, `W_K` e `W_V` de dimensão `[512, 64]` e aplica o Scaled Dot-Product Attention sem mascara causal, pois o Decoder deve acessar toda a frase do Encoder.

A saída tem forma `[1, 4, 64]`: para cada token gerado pelo Decoder, um vetor de contexto condensado a partir dos 10 tokens do Encoder.

## Como executar

```bash
python decoder.py
```

## Dependências

- Python 3.x
- NumPy
