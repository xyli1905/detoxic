# tag: R1D1d0-L1 -> 1-layer unidirectional RNN with dropout rate 0 on output plus 1 linear layer as classifier

# 1 rnn layer + bi-directional
/Users/xyli1905/anaconda3/bin/python train_baseline.py \
--classifier_net LSTM \
--tag R1D2d0-L1 \
--is_bidirectional True \
--max_epoch_C 1

# 2 rnn layer + unidirectional
/Users/xyli1905/anaconda3/bin/python train_baseline.py \
--classifier_net LSTM \
--tag R2D1d0-L1 \
--number_layers 2 \
--max_epoch_C 1

# 1 rnn layer + unidirectional
/Users/xyli1905/anaconda3/bin/python train_baseline.py \
--classifier_net LSTM \
--tag R1D1d0-L1 \
--max_epoch_C 1

# 2 rnn layer + bi-directional
/Users/xyli1905/anaconda3/bin/python train_baseline.py \
--classifier_net LSTM \
--tag R2D2d0-L1 \
--number_layers 2 \
--is_bidirectional True \
--max_epoch_C 1