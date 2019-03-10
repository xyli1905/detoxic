# tag: R1D1d0-L1 -> 1-layer unidirectional RNN with dropout rate 0 on output plus 1 linear layer as classifier
# 1 rnn layer + bi-directional

# round 1
/Users/xyli1905/anaconda3/bin/python train_baseline.py \
--model_type balanced \
--classifier_net LSTM \
--tag R1D2d0_2-L1 \
--is_bidirectional True \
--dropout_rate 0.2 \
--max_epoch_C 20

# round 2
/Users/xyli1905/anaconda3/bin/python train_baseline.py \
--model_type balanced \
--classifier_net LSTM \
--tag R1D2d0_2-L1 \
--is_bidirectional True \
--dropout_rate 0.2 \
--max_epoch_C 40 \
--load_epoch_C 20

# round 3
/Users/xyli1905/anaconda3/bin/python train_baseline.py \
--model_type balanced \
--classifier_net LSTM \
--tag R1D2d0_2-L1 \
--is_bidirectional True \
--dropout_rate 0.2 \
--max_epoch_C 60 \
--load_epoch_C 40

# round 4
/Users/xyli1905/anaconda3/bin/python train_baseline.py \
--model_type balanced \
--classifier_net LSTM \
--tag R1D2d0_2-L1 \
--is_bidirectional True \
--dropout_rate 0.2 \
--max_epoch_C 80 \
--load_epoch_C 60

# round 5
/Users/xyli1905/anaconda3/bin/python train_baseline.py \
--model_type balanced \
--classifier_net LSTM \
--tag R1D2d0_2-L1 \
--is_bidirectional True \
--dropout_rate 0.2 \
--max_epoch_C 100 \
--load_epoch_C 80