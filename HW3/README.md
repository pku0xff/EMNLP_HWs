- Dependencies

  1. Pretrained word embedding `GloVe` (100d).
  2. `spacy`: to do dependency parsing.
  3. `sklearn`: to vectorize features and calculate metrics.
  4. Others: `numpy`, `torchtext`, `torch`

- Structure

  Code:

  - `trigger_detection.py`, `trigger_classification.py`, `argument_identification.py` and `argument_classification.py` are the 4 modules. The models are trained separately.
  - `run.py` is the pipeline which integrates the modules and do inference.

  Data:

  - Raw data is in the directory `data`.
  - Trained models are saved as `trigger_detection.npy`, `trigger_classification.pt`, `argument_identification.npy` and `argument_classification.npy`.