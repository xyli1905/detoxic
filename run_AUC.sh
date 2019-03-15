
PYTHON="/Users/xyli1905/anaconda3/bin/python"

# ------------------------- results for threshold ------------------------ #
range="0.003 0.01 0.03 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.96"
# # range="0.6 0.7 0.8 0.9 0.96"

opt_epoch="11"

for i_threshold in $range
do
echo "run for epoch: $i_threshold"
$PYTHON test.py \
--model_type baseline \
--classifier_net LSTM \
--tag R1D2d0_2-L1 \
--is_bidirectional True \
--dropout_rate 0.2 \
--load_epoch_C $opt_epoch \
--threshold $i_threshold
done
# --model_type baseline \
# --classifier_net LSTM \
# --tag R2D1d0-L1 \
# --number_layers 2 \
# --load_epoch_C 11 \
# --threshold $i_threshold
# ------------------------- results for threshold ------------------------ #