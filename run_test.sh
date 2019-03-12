
PYTHON="/Users/xyli1905/anaconda3/bin/python"

# ------------------------- results for epochs ------------------------ #
# range="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
# range="1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100"

# for i_epoch in $range
# do
# echo "run for epoch: $i_epoch"
# $PYTHON test.py \
# --model_type balanced \
# --classifier_net LSTM \
# --tag R1D2d0_2-L1 \
# --is_bidirectional True \
# --dropout_rate 0.2 \
# --load_epoch_C $i_epoch
# done
# round 1
# --is_bidirectional True \
# --dropout_rate 0.2 \
# ------------------------- results for epochs ------------------------ #


# ------------------------- results for threshold ------------------------ #
range="0.003 0.01 0.03 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.96"
for i_threshold in $range
do
echo "run for epoch: $i_epoch"
$PYTHON test.py \
--model_type baseline \
--classifier_net LSTM \
--tag R1D1d0-L1 \
--load_epoch_C 9 \
--threshold $i_threshold
done