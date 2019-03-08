# tag: R1D1d0-L1 -> 1-layer unidirectional RNN with dropout rate 0 on output plus 1 linear layer as classifier
/Users/xyli1905/anaconda3/bin/python train_baseline.py \
--classifier_net LSTM \
--tag R1D1d0-L1 \
--max_epoch_C 1

/Users/xyli1905/anaconda3/bin/python train_baseline.py \
--classifier_net LSTM \
--tag R1D1d0_2-L1 \
--dropout_rate 0.2 \
--max_epoch_C 1
