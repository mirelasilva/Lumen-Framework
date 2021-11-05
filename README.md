# Lumen Framework
This repository contains the source code, dataset, and Jupyter notebooks used towards the Lumen framework. At this stage, we kindly ask that this repository not be distributed or disseminated to others.


# Lumen code setup

## python env & packages

- python version

  - `3.6.6`

- core package versions

  - ```
    gensim==3.8.1
    Keras==2.3.1
    nltk==3.4.5
    pandas==1.1.5
    scikit-learn==0.21.3
    tensorflow==2.0.0
    ```

- Note that `nltk` will also need additional subpackages, etc.

  - see the

  - ```
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    ```

- Other package versions can be found in the `requirements_lumen366.txt` file

## dataset path/location

- Note that

  - all the path is the `relative path` to `the root of the project`

- raw lumen dataset
- data\s2021_lumen_raw_data
- cleaned lumen dataset
  - data\s2021_lumen_clean_data
- liwc dataset should be unzipped to
  - data\tmp_liwc
- glove dataset should be unzipped to
  - data\tmp_glove

## notebooks for experiment

### data cleanup

- run `s01 02 01 2021-05-23 exploratory analysis.ipynb`
  - clean up the data
  - create `data\s2021_lumen_clean_data\s2021_05_23_01_lumen_clean_data.csv`
- run `s02 01 2021-05-23 clean data for topic model.ipynb`
  - generate text which can be used for `topic modeling`
  - create `data\s2021_lumen_clean_data\s2021_05_23_02_lumen_clean_doc_data.csv`
- run `s02 02 2021-05-23 create_feature_sia_liwc.ipynb`
  - add `sia` and `liwc` features to the data table
  - create `data\s2021_lumen_clean_data\s2021_05_23_03_lumen_clean_doc_sia_liwc.csv`
- run `s02 03 2021-06-20 create_feature_sia_liwc for classification.ipynb`
  - put inputs and output in one data table, make it easier for model building & evaluation
  - create `data\s2021_lumen_clean_data\s2021_06_20_01_lumen_clean_doc_sia_liwc_classify.csv`

### experiment

- run `s03 01 2021-06-20 feature_sia_liwc_classification_analysis.ipynb`
  - statistic analysis of the lumen dataset
- `s04 01 2021-06-20 run_Lda_liwc_sia_RandomForest_real_doc_cv_gs.ipynb`
  - use grid-search cross-validation to obtain the best parameter: number of topics for the topic modeling, number of estimators in the random forest
- `s04 02 2021-06-20 run_Lda_liwc_sia_RandomForest_real_real_doc_feature_importance.ipynb`
  - with the parameter obtained from grid-search cross-validation
  - experiment with our main `Lumen` framework
- `s05 01 2021-06-20 classification_simple_neural_network.ipynb`
  - experiment with a `simple neural network` method
- `s05 02 2021-06-20 classification_LSTM.ipynb`
  - experiment with a `simple Long short-term memory (LSTM)` method
- `s05 03 2021-06-20 classification_stack_LSTM.ipynb`
  - experiment with a ` stacked Long short-term memory (LSTM)` method
- `s06 01 2021-06-30 run_LLda_real_doc_cv.ipynb`
  - experiment with ` Labeled-lda` method
- `s06 02 01 2021-06-30 classification_bidirectional_LSTM_cv_glove50.ipynb`
  - experiment with ` bidirectional LSTM` method with 50-dimension `word-embeding`

## end
