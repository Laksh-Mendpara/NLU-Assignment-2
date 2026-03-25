# Character-Level Name Generation Report

Dataset: `q2/dataset/data.txt`  
Training names: `1000` unique names  
Vocabulary: `28` symbols including `<pad>`, `<bos>`, `<eos>`  
Preprocessing: lowercasing plus whitespace cleanup. One malformed line in the raw file contained internal tabs, so internal whitespace was removed during loading.

All three systems below are **character-level sequence models**. Each training example is a character prefix beginning with `<bos>`, and the target is the next character or `<eos>`. During generation, the model starts from `<bos>` and samples one character at a time until `<eos>`.

## Task 1: Model Implementation

Shared hyperparameters across all models:

- Embedding size: `32`
- Hidden size: `64`
- Number of layers: `1`
- Learning rate: `0.003`
- Batch size: `64`
- Epochs: `20`
- Dropout: `0.15`
- Sampling temperature: `0.85`
- Generated samples for evaluation: `300` names per model

| Model | Character-level architecture | Trainable parameters |
| --- | --- | ---: |
| Vanilla RNN | Character embedding -> vanilla `RNN` -> final hidden state -> linear layer over character vocabulary | 8,988 |
| BLSTM | Character embedding -> bidirectional `LSTM` -> concatenate forward and backward final states -> linear layer over character vocabulary | 54,684 |
| RNN + Attention | Character embedding -> vanilla `RNN` -> additive attention over all prefix hidden states using the final hidden state as the query -> concatenate context and query -> linear layer over character vocabulary | 19,165 |

Best training losses after 20 epochs:

- Vanilla RNN: `1.7044`
- BLSTM: `1.3852`
- RNN + Attention: `1.6474`

## Task 2: Quantitative Evaluation

Metrics:

- Novelty Rate = percentage of generated names not present in the training set
- Diversity = number of unique generated names divided by total generated names

| Model | Novelty Rate | Diversity | Avg. Generated Length |
| --- | ---: | ---: | ---: |
| Vanilla RNN | 88.67% | 0.963 | 6.56 |
| BLSTM | 58.67% | 0.900 | 6.35 |
| RNN + Attention | 86.00% | 0.970 | 6.40 |

Comparison:

- **Best novelty:** Vanilla RNN
- **Best diversity:** RNN + Attention
- **Lowest training loss / most memorization:** BLSTM

Interpretation:

- The `BLSTM` fits the training distribution most strongly, but that comes with much lower novelty and diversity, which suggests heavier memorization of seen names.
- The `Vanilla RNN` gives the best novelty while still generating many plausible names.
- The `RNN + Attention` keeps novelty close to the vanilla model and gives the best diversity, so it offers the strongest balance between variety and realism in this experiment.

## Task 3: Qualitative Analysis

### Vanilla RNN

- Realism: Often produces plausible Indian-style names or close variants, especially for common prefixes and endings.
- Common failure modes: copied training names, occasional early stopping, and some stitched endings such as `ramakam` or `sonadesi`.
- Representative samples: `shasam`, `sankas`, `vitesh`, `radhsura`, `saheet`, `raghivan`, `lakhila`, `ramakam`, `madhakar`, `kesali`, `sonadesi`, `dhuran`

### BLSTM

- Realism: This model often gives the smoothest phonetic flow and the most name-like outputs.
- Common failure modes: strongest memorization, more duplicate generations, and many direct reproductions of training names such as `madhav`, `vijendra`, `anuj`, and `vidya`.
- Representative samples: `manjo`, `ramak`, `prabhavti`, `amabi`, `ranju`, `chandrakant`, `jayakesh`, `yusman`, `bipal`, `bimpa`, `madhav`, `savanya`

### RNN + Attention

- Realism: Produces varied and mostly natural-looking names, and attention seems to help preserve useful prefix patterns.
- Common failure modes: occasional repeated characters or over-assembled names such as `balweeeet`, `sutunta`, and `shyapunta`.
- Representative samples: `dipender`, `jatil`, `sangeeta`, `mukesh`, `anild`, `sutunta`, `buldev`, `balweeeet`, `garvit`, `jagadip`, `shyapunta`, `vasanta`

## Overall Conclusion

- If the goal is **maximum novelty**, the `Vanilla RNN` performed best.
- If the goal is **best balance between novelty and diversity**, the `RNN + Attention` performed best.
- If the goal is **most fluent but least novel outputs**, the `BLSTM` performed best, but it also copied the training set much more often.

Artifacts produced in `q2/results`:

- `results.json`: raw metrics, losses, and sample pools
- `vanilla_rnn_samples.txt`
- `blstm_samples.txt`
- `attention_rnn_samples.txt`
