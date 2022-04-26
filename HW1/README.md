- Dependencies

  In addition to the original data, a lemmatization list from https://github.com/michmech/lemmatization-lists is needed before run the shell script.

- Structure

  Files:

  - The `data` directory contains source data (`sst_train.csv`, `sst_test.csv`, `yelp_train.csv`, `yelp_test.csv`) and preprocessed data (in directory `sst` and `yelp`)
    Only the lemmatization list is preserved when submitting the HW.
  - Training logs and testing results are in `result` directory. These are the data I used to write my report.

  Code:

  - `preprocess.py` preprocess raw texts and save the results in `data` directory. 
  - `run.py` read the preprocessed data and do feature extraction and model training as well as testing.
  - Models are implemented in `run.py` as python `Class`es, including training, validating and testing process.

- To do all experiments from preprocessing to testing, type `sh run.sh` in the shell.

