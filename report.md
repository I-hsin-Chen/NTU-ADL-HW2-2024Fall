# ADL HW2 Report

r12944005 陳乙馨

## Model

### Model

T5 模型是一個可以應用於許多 Natural Language Generation (NLG) 下游任務的預訓練語言模型、例如 : Translation, Captioning 以及Summarization 等等，不同於 BERT 僅有 encoder，T5 擁有 encoder-decoder 的核心架構、使其可以完成 Sequence to sequence 的任務，encoder 由多層 self-attention 和 Feed-Forward Neural Network 組合而成，decoder 除了使用 self-attention 之外、還有與 encoder 的 cross attention 以幫助其生成與文本更有關聯的輸出，此次作業使用的 mt5 則是專為處理多語言 ( multilingual ) 任務而設計。

該模型 Sequence to sequence 的特性使他天生就可以很好的適應 text-summarization 這個下游任務，除了需給予訓練資料 : 文本與摘要 ( 即 ground truth )，為了提升他的收斂速度以及訓練效率，我也在本次作業訓練的過程中 加上 `source_prefix` 的參數 `"summarize : "` 給予模型一個明確的任務提示。

### Preprocessing

本次作業中我參考了 [huggingface sample code](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization_no_trainer.py) 的作法，前處理的過程中，我進行了以下幾個步驟 : 

- **Text / Summary Mapping**：根據 dataset 的欄位名稱，確認要處理的輸入欄位 ( `maintext` ) 和摘要欄位 ( `title`)，確保模型能正確讀取文本與其對應的摘要。

- **Truncation / Padding**：對於輸入文本與摘要，使用 `tokenizer` 進行 tokenization，並根據最大長度參數 (`max_length`) 將文本進行截斷，確保輸入符合模型的長度限制。

- **Ignore Padding Token For Loss**：在對摘要進行 tokenization 時，為了使長度一致也會進行 padding，然而在計算 loss 時需要忽略這些 padding 的部分，因此需要將 padding 部分 (`pad_token_id`) 替換為 -100，以便模型不將這些部分計入損失函數。

## Training

### Hyperparameter

以下是我表現最好的模型參數 : 

| model                           | `google/mt5-small` |
| ------------------------------- | ------------------ |
| **num_train_epochs**            | 20                 |
| **num_warmup_steps**            | 500                |
| **max_source_length**           | 512                |
| **max_target_length**           | 128                |
| **per_device_train_batch_size** | 2                  |
| **gradient_accumulation_steps** | 2                  |
| **learning_rate**               | 1e-4               |
| **lr_scheduler_type**           | linear             |
| **optimization**                | AdamW              |
| **num_beams**                   | 10                 |

這個 model 在 `tw-rouge` 的 f1-score 表現如下 : 

| rouge-1 | rouge-2 | rouge-l |
| ------- | ------- | ------- |
| 0.26851 | 0.10809 | 0.23965 |

* 在選擇 learning rate 時我閱讀了這串[討論](https://discuss.huggingface.co/t/t5-finetuning-tips/684)，裡面提及 1e-4 和 3e-4 是比較穩定的學習率。
* 因為我的運算資源有限、gpu 也沒有很好，因此我選擇用較小的 batch size 換取較大的 max_source_length ( 得以讓模型處裡更完整的文本 )，且搜尋網路上大家微調 T5 的經驗幾乎都是使用 512。
* 以上這組參數花費了近 8 小時的訓練時間。至於如何決定 epoch 的數量，我一開始先設定為 5 並不斷往上加 5，發現到 epoch = 15 時 performance 的提升就趨緩了，最後選擇了 epoch=20。

### Learning Curves

![](https://github.com/I-hsin-Chen/NTU-ADL-HW2-2024Fall/blob/main/curve.png)

## Generation Strategies

### Strategies

#### **Greedy Search**

相當於設定 num_beams = 1，在每一步生成時，模型會選擇當前步驟中概率最高的字詞作為輸出。這代表它總是選擇最可能的詞彙而不考慮未來的可能性，因此可能會卡在局部最優解，導致生成結果不夠多樣或流暢。

#### **Beam Search**

Beam Search 會在每一步生成過程中持續追蹤複數個候選的 sequences (paths) 而不是僅選擇一個最可能的詞彙。生成結束時最終輸出的是具有最高總概率的 sequence。當 num_beams 很小，雖然計算複雜度較低，但也很有可能遇到 chit-chat Dialogues (複誦一樣的單詞)，例如在我的 output 中就有出現 `全台最熱門鐵道自行車!\n三義「三義舊山線鐵道自行車」,還有龍騰斷橋、龍騰斷橋、龍騰斷橋、龍騰斷橋` 的結果，num_beams 越大、越有可能找到合適的答案，可是同時計算複雜度也更高。

#### **Top-k Sampling**

Top-k Sampling 主要是在生成過程中加入一些隨機性，在每一步生成時，模型會從概率最高的 `k` 個詞彙中隨機選擇一個作為下一個字詞。這樣可以避免模型總是選擇概率最高的字詞從而產生更多變化和創意，然而 `k` 的大小有時很難以拿捏，因為每個生成步驟的機率分佈都不相同，因此後來又衍伸出了 Top-p Sampling 這個方法。

#### **Top-p Sampling**

Top-p Sampling 根據累積概率來決定候選的詞彙多寡 ( 有點像是動態的調整 k )。在生成過程中，模型會選擇使累積概率達到門檻值 `p` 的前幾個詞彙，然後從這個集合中隨機抽取下一個字詞。這使得生成的序列在保持多樣性的同時也保持了一定的語義連貫性。

#### **Temperature**

Temperature 跟以上選擇哪個詞彙的策略比較不同，他會直接影響字詞的機率分佈。較高的 Temperature 會使模型生成的分佈更加平坦（即更隨機），而較低的值會使分佈更加陡峭（即更傾向於選擇概率高的詞彙）。

### Hyperparameters

| Strategy          | parameters                  | rouge-1 | rouge-2 | rouge-l |
| ----------------- | --------------------------- | ------- | ------- | ------- |
| Greedy            | `num_beams=1`               | 0.25397 | 0.09346 | 0.22671 |
| Beam              | `num_beams=5`               | 0.26801 | 0.10657 | 0.23954 |
| Beam              | `num_beams=10`              | 0.26851 | 0.10809 | 0.23965 |
| Top-k             | `top_k=50,temperature=1`    | 0.20982 | 0.06799 | 0.18445 |
| Top-k             | `top_k=100,temperature=1`   | 0.20224 | 0.06670 | 0.17828 |
| Top-p             | `top_p=0.85,temperature=1`  | 0.20633 | 0.07034 | 0.18276 |
| Top-p             | `top_p=0.9,temperature=1`   | 0.19826 | 0.06626 | 0.17609 |
| Top-p/Temperature | `top_p=0.9,temperature=0.5` | 0.25017 | 0.09138 | 0.22269 |
| Top-p/Temperature | `top_p=0.9,temperature=2`   | 0.02293 | 0.00133 | 0.02120 |

#### Observation

* 最後表現最佳的策略是 `num_beams=10`，沒有使用任何影響隨機性的參數 ( 即`top-k`, `top-p`, `temperature` )，試著推敲背後可能的原因，我想是因為 summarization 是一個比較 deterministic 的任務，不太需要生成結果具有創意或是多樣性。
* 當固定 `top_p=0.9` 時，可以發現 `temperature=0.5` 的表現比 `temperature=1` 好很多，`temperature=2` 的結果則是所有策略最糟糕的，前面提及過 `temperature` 越低會使字詞的概率分布越陡峭，也就會更可能選到概率較高的詞彙，從此結果也可以推論出隨機性越低時表現是越好的。

## Reference

[ChatGPT](https://chatgpt.com/) Mainly for code comprehensiona and understanding strategies

[Github Transformer : run_summarization_no_trainer.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization_no_trainer.py)

[CSDN : Generation Strategies](https://blog.csdn.net/muyao987/article/details/125917234)

[HuggingFace : Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)

[HuggingFace : T5 Finetuning Tips](https://discuss.huggingface.co/t/t5-finetuning-tips/684/6)

[T5 : a detailed explanation](https://medium.com/analytics-vidhya/t5-a-detailed-explanation-a0ac9bc53e51)
