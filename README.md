# DeToxic
This is a class project in CMPS240 at UCSC, aiming to detect toxic content in Quora questions using state-of-art NLP techniques. This project is from an on-going [Kaggle competition](https://www.kaggle.com/c/quora-insincere-questions-classification), which provides us all the datasets and pretrained embeddings.

Related topics: *NLP, classification, Deep Learning, content & style, PyTorch, Google Cloud Platform, GPU, Kaggle*

## Structure
We only mention the function of key directories here, leaving the diagram of the whole structure of the source code at the end of this README.md file:
* `./checkpoints`: save the trained weight and optimizer files
* `./data_proc`: python script for data preprocessing; save the processed data
* `./model`: contain classes that combine networks to build models, e.g. baseline
* `./networks`: contain classes that define networks, e.g. LSTM
* `./options`: contain the class define the options in our model
* `./utils`: contain classes for util functions

## How to train the model
For example,
```
python train-baseline.py \
--model_type baseline \
--classifier_net LSTM \
--number_layers 2 \
--is_bidirectional True \
--tag R2D2d0-L1 \
--max_epoch_C 20 \
--load_epoch_C 10 \
--threshold 0.5
```
* In this example we continue to train a specified LSTM model, starting from epoch 10 to max epoch 20;
  * one must make sure the existence of pretrained weight file and optimizer file for epoch 10 in the checkpoints directroy.
* `--model_type baseline` means the intermedeiate files will be saved in `./checkpoints/baseline/`;
* `--classifier_net LSTM --number_layers 2 --is_bidirectional True` specify the architecture of the LSTM model: two bidirectional LSTM layers;
* `--tag R2D2d0-L1` defines the tag field in the name of weight and optimizer files, e.g. `net_LSTM_R2D2d0-L1_1_id.pth` and `opt_LSTM_R2D2d0-L1_1_id.pth`;
  * here `R` refers to the number of RNN (LSTM) layers; `D` refers to the number of directions; `d` refers to the dropout rate; and `L` refers to the number of fully-connected layers in the classifier.
* `--threshold 0.5` set the threshold for binary classification, i.e. if the predicted probability is greater than 0.5, it will be assigned to the possitive class;
* All possible options are defined in the file `./options/base_options.py`.


## How to evaluate the modle (on validation set)
* One may use command line directly,
  ```
  python test.py \
  --model_type baseline \
  --classifier_net LSTM \
  --number_layers 2 \
  --is_bidirectional True \
  --tag R2D2d0-L1 \
  --load_epoch_C 10 \
  --threshold 0.5
  ```
  * the options are very similar to the ones used in the training phase, basically those options explicitly describe the model for testing;
  * in this case, before test the model LSTM(R2D2d0-L1), one must make sure the existence of file `./checkpoints/baseline/net_LSTM_R2D2d0-L1_10_id.pth` (not the opt file is not necessary for testing).
* or simply run the shell script (need to modify the shell script accordingly)
  ```bash
  bash run_test.sh
  ```

## Appendix
The whole structure of the source code is listed in the following tree diagram:
```
.
├── README.md
├── checkpoints
│   ├── balanced
│   ├── baseline
│   └── triplet
├── data_proc
│   ├── __init__.py
│   ├── embedding_preprocess.py
│   ├── preprocess.ipynb
│   ├── preprocessor_demo.ipynb
│   ├── processed_data
│   └── question_preprocess.py
├── debug
├── model
│   ├── __init__.py
│   ├── baseline.py
│   ├── basemodel.py
│   └── triplet.py
├── networks
│   ├── GRU.py
│   ├── LR.py
│   ├── LSTM.py
│   ├── __init__.py
│   ├── __pycache__
│   ├── basenetwork.py
│   ├── embedding.py
│   └── linear.py
├── options
│   ├── __init__.py
│   ├── __pycache__
│   └── base_options.py
├── results
├── run_AUC.sh
├── run_test.sh
├── test.py
├── train_baseline.py
├── train_triplet.py
└── utils
    ├── __init__.py
    ├── __pycache__
    ├── dataloader.py
    └── util.py
```
