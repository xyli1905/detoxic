
PYTHON="/Users/xyli1905/anaconda3/bin/python"

# ------------------------- results for epochs ------------------------ #
# range="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
range="21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40"
# range="1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100"

for i_epoch in $range
do
echo "run for epoch: $i_epoch"
$PYTHON test.py \
--model_type baseline \
--classifier_net LSTM \
--tag R1D2d0_2-L1 \
--is_bidirectional True \
--dropout_rate 0.2 \
--load_epoch_C $i_epoch
done
# --model_type balanced \
# --classifier_net LSTM \
# --tag R1D2d0_2-L1 \
# --is_bidirectional True \
# --number_layers 2 \
# --dropout_rate 0.2 \
# --load_epoch_C $i_epoch
# ------------------------- results for epochs ------------------------ #