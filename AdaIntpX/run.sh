## M
root_path=$your_data_path
datasets=(ELC) # Traffic Weather ELC Exchange ETTm2 ETTm1 ETTh1 ETTh2 Solar
# when you run FreTS or PatchTST, it may case out-of-memory, plase set small batch size.
models=(iTransformer)  # iTransformer DeepBooTS FreTS FourierGNN DLinear Periodformer Informer  FEDformer
pred_lens=(96) # 192 336 720
kappas=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8  0.9  0.95) # 0.2 0.3 0.4 0.5 0.6 0.7 0.8  0.9 0.95 )
for dataset in ${datasets[*]}; do
for pred_len in ${pred_lens[*]}; do
for model in ${models[*]}; do
for kappa in ${kappas[*]}; do

CUDA_VISIBLE_DEVICES=3 python -u run.py \
  --is_training 1 \
  --root_path $root_path \
  --model_id $dataset'_96_'$pred_len \
  --model $model \
  --data $dataset \
  --seq_len 96 \
  --train_epochs 10 \
  --pred_len $pred_len \
  --batch_size 16 \
  --kappa $kappa \
  --features M \
  --itr 1 #>logs/$model'_'$dataset'_96_'$pred_len'_kappa_'$kappa.log

done
done
done
done