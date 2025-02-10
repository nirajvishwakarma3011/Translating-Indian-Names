# Transliteration-Using-Sequence-to-Sequence-Modelling

Neural language models are able to successfully capture patterns across Indian names. This project extend upon that idea to learn conditional language models for the task of transliteration: converting Indian names in the English alphabet to Hindi.
```
└─── Translation-Using-Sequence-to-Sequence-Modelling
     ├─── main.py
     ├─── src-tokenizer
     │    └─── tokenizer.pkl
     ├─── tgt-tokenizer
     │    └─── tokenizer.pkl
     ├─── rnn.enc-dec
     │    ├─── model.pt
     │    ├─── loss.json
     │    ├─── outputs.csv
     │    └─── metadata.json
     └─── rnn.enc-dec.attn
          ├─── model.pt
          ├─── loss.json
          ├─── outputs.csv
          └─── metadata.json
```
# Tokenization
We first prepare tokenization strategy for feeding name pairs as a sequence to different models. For English this could be as simple as using individual characters as tokens, but Hindi has accents (मात्राएँ), a larger set of vowels (स्वर), consonants (व्यंजन), and additional composition rules (half-letters, etc.), so such a simple strategy may not be effective.

In NLP literature, multiple strategies exist for automatically learning a suitable sub-word tokenization strategy from the given data. Such tokenizers exist in two types:

- Given a set of initial tokens, learn suitable combinations which are added as new tokens until a certain vocabulary size is reached. Examples of these include BPE Tokenization and WordPiece Tokenization, introduced by the BERT paper.
- Given a large set of initial tokens, learn suitable rules to reduce the size of the vocabulary to a desired size. An example of this includes SentencePiece Tokenization.
- Given empirical results, these are popular strategies to learn tokenization automatically from given data.

We can have a tokenizer that operates jointly over both languages or have separate tokenizers for English and Hindi.
Tokenizer can learn the tokenization from data (using any one of the techniques mentioned above) or can use a fixed set of rules for decomposition.

The tokenizer will learn a mapping of tokens to ids and vice versa and use these to map strings. This mapping can be built based on merge rules (BPE, WordPiece, etc.) or hand-crafted rules, in the Tokenizer.train() function. Additionally the tokenizer will also handle preprocessing and postprocessing of strings during the encoding phase (string to tokens).

<img width="794" alt="Screenshot 2024-06-28 at 11 41 52 PM" src="https://github.com/pritamgouda11/Translation-Using-Sequence-to-Sequence-Modelling/assets/46958858/59242a0e-7a6a-4bab-84d8-e047962013f1">

<img width="650" alt="Screenshot 2024-06-28 at 11 42 10 PM" src="https://github.com/pritamgouda11/Translation-Using-Sequence-to-Sequence-Modelling/assets/46958858/6e42e969-293f-4028-beaf-2a1d8d97a4dc">

# Model-Agnostic Training

Next, we implement a Trainer to train different models, since the data and tokenizer remains the same for all models. This trainer will receive the model, a loss function, an optimizer, a training and (optionally) a validation dataset and use these to train (and validate) the model. The trainer will also take care of handling checkpoints for training, which can be used to resume training across sessions. Derived classes can also be defined to handle different architectures.

# Seq-2-Seq Modeling with RNNs

![image](https://github.com/pritamgouda11/Translation-Using-Sequence-to-Sequence-Modelling/assets/46958858/54d89610-ded5-4032-a7c9-34f60a4882c9)

An encoder-decoder network using RNNs is implemented, to learn a conditional language model for the task of translating the names to Hindi. We can use any type of RNN for this purpose: RNN, GRU, LSTM, etc. Consult the pytorch documentation for additional information.

**Additional tips for training:**
  - Use regularization: Dropout, etc.
  - Use a suitable optimizer, such as Adam.
  - Format data accordingly before passing it to the trainer, using the helper functions.
  - Do you need to pad sequences when processing inputs as a batch?
    
# Seq-2-Seq Modeling with RNN + Attention
Augment the Encoder-Decoder architecture to utilize attention, by implementing an Attention module that attends over the representations / inputs from the encoder. Many approaches have been proposed in literature towards implementing attention. Some popular approaches are desribed in the original paper by Bahdanau et al., 2014 on NMT and an exploratory paper by Luong et al, 2015 which explores different effective approaches to attention, including global and local attention.

# Evaluation

- Accuracy: From a parallel corpus, number of translations the model got exactly right. Higher the better. Note that this makes sense only for this task. and lacks granularity.
- Edit Distance: Number of edits at the character level (insertions, deletions, substitutions) required to transform your model's outputs to a reference translation. Lower the better.
- Character Error Rate (CER): The rate at which your system/model makes mistakes at the character level. Lower the better.
- Token Error Rate (TER): The rate at which your system/model makes mistakes at the token level. Lower the better. Depending on your tokenizer implementation, could be the same as CER.
- BiLingual Evaluation Understudy (BLEU): Proposed by Papineni et al., 2002, BLEU is a metric that assess the quality of a translation against reference translations through assessing n-gram overlap. Higher the better.
Since accents and half-letters exist as separate characters in the Unicode specification, and can change the interpretation of the output, metrics that operate at the character level will treat these separately.
