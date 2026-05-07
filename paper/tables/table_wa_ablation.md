# Table 4 — $W_a$ decomposition (Exp M)

| Task | LatentMAS (trained $W_a$) | Identity $W_a$ | No transfer |
|---|---|---|---|
| GSM8K | 92.0 (n=500) | 91.0 (n=100) | 90.8 (n=500) |
| ARC-Challenge | 92.6 (n=500) | 91.0 (n=100) | 93.2 (n=500) |
| MBPP+ | 73.3 (n=378) | 74.0 (n=100) | 67.5 (n=378) |

*Identity $W_a$* removes the trained linear bridge while keeping the KV channel intact. *No transfer* removes the channel entirely. The trained $W_a$ adds 0–1pp on GSM8K/ARC and +5.8pp on MBPP+ (McNemar p=0.010 vs no-transfer): the alignment matrix is load-bearing on code generation but contributes little on arithmetic and multiple choice.
