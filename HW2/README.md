- Dependencies

  - Pretrained word embedding `GloVe`, which can be downloaded via the toolkit `torchtext`.

- Structure

  Files:

  - The `data`  directory contains source data.
  - The `.vector_cache` directory will be created in running to cache word embeddings.
  - `LinearModel_log.txt` and `RNN_log.txt`: record the training stage and testing results.

  Code: model structures are implemented as class. Training and testing process are implemented as functions.

  - `run_RNN.py`: build the RNN structure, train it and test it.
  - `run_LinearModel.py`: build the Linear Model, train it and test it.
  - `analysis.py`: analysis the wrongly predicted tokens.

