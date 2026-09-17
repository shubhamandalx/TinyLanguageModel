# data/

Place your training text corpus here as:

```
data/input.txt
```

(the path is configurable via `GPTConfig.input_file` in `config.py`)

The file should be a single plain-text, UTF-8 encoded file. It is
tokenized as one continuous stream and used for next-token
prediction: the model is trained to predict each token from the
tokens that came before it.

No other preprocessing is required — tokenization happens
automatically via `tiktoken` when `train.py` runs.

Here i have TinyStories Validation file for demo purposes.