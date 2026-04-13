## M
root_path=$your_data_path
datasets=(Weather) # Traffic Weather ELC Exchange ETTm2 ETTm1 ETTh1 ETTh2 Solar
# when you run FreTS or PatchTST, it may case out-of-memory, plase set small batch size.
models=(iTransformer)  # iTransformer DeepBooTS FreTS FourierGNN DLinear Periodformer Informer  FEDformer
pred_lens=(96) # 192 336 720
for dataset in ${datasets[*]}; do
for pred_len in ${pred_lens[*]}; do
for model in ${models[*]}; do

CUDA_VISIBLE_DEVICES=5 python -u run_xy_mul.py \
  --is_training 1 \
  --root_path $root_path \
  --model_id $dataset'_96_'$pred_len \
  --model $model \
  --data $dataset \
  --seq_len 96 \
  --train_epochs 10 \
  --pred_len $pred_len \
  --batch_size 16 \
  --delta 0.2 \
  --itr 1 >logs_xy/$model'_'$dataset'_96_xy_mul_'$pred_len.log

done
done
done