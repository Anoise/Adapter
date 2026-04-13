## M
root_path=/harbor-data/daojun/Data/TS_Small
datasets=(ETTm1 ETTh1 ETTh2 electricity exchange_rate traffic weather)
# when you run FreTS or PatchTST, it may case out-of-memory, plase set small batch size.
models=(Informer Autoformer FEDformer Periodformer FourierGNN DLinear Transformer) 
pred_lens=(96 192 336 720)
for dataset in ${datasets[*]}; do
for model in ${models[*]}; do
for pred_len in ${pred_lens[*]}; do

CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --data_path $dataset'.csv' \
  --model_id $dataset'_96_'$pred_len \
  --model $model \
  --data $dataset \
  --seq_len 96 \
  --pred_len $pred_len \
  --itr 1 >logs/$model'_'$dataset'_96_'$pred_len.log

done
done
done