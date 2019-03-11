
PYTHON="/Users/xyli1905/anaconda3/bin/python"
range="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

for i_epoch in $range
do
echo "run for epoch: $i_epoch"
$PYTHON test.py \
--model_type baseline \
--classifier_net LSTM \
--tag R1D1d0-L1 \
--load_epoch_C $i_epoch
done
# round 1
# --is_bidirectional True \
# --dropout_rate 0.2 \


