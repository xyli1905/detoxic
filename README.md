# DeToxic
This is a class project in CMPS240 at UCSC, aiming to detect toxic content in Quora questions. This project is from an on-going [Kaggle competition](https://www.kaggle.com/c/quora-insincere-questions-classification), which provides us all the datasets and pretrained embeddings.

Related topics: *NLP, classification, Deep Learning, content & style, PyTorch, Google Cloud Platform, GPU, Kaggle*

## Stage 1
### learning objectives
* Word2Vec
* basic RNN, LSTM, GRU

# tested models
* baseline
	* ~~R1D1d0-L1~~
	* ~~R1D1d0_2-L1~~
	* ~~R1D2d0-L1~~
	* ~~R2D1d0-L1~~
	* ~~R2D2d0-L1~~
	* EmbLR
	* BoW
* balanced
	* ~~R1D2d0_2-L1~~


# test for threshold:
* baseline
    * R1D1d0-L1: epoch 9, epoch 15
    * R1D1d0_2-L1: epoch 15
    * R1D2d0-L1: epoch 11
    * R2D1d0-L1: epoch 11
    * R2D2d0-L1: epoch 14
    * EmbLR: epoch 12
    * BoW: 
* balanced
    * R1D2d0_2-L1