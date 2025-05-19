if [ ! -d "./train_info/Traffic" ]; then
    mkdir ./train_info/Traffic
fi

if [ ! -d "./_wts/Traffic" ]; then
    mkdir ./_wts/Traffic
fi

if [ ! -d "./figure/Traffic" ]; then
    mkdir ./figure/Traffic
fi

root_path_name=../_dat/
random_seed=3407
data='Traffic'
look_back_length=96

for prediction_length in 12 24 48
do
    python -u main.py \
        --random_seed $random_seed \
        --data $data \
        --root_path $root_path_name \
        --features 'M' \
        --look_back_length $look_back_length \
        --label_len $look_back_length \
        --prediction_length $prediction_length \
        --decomposition 1 \
        --kernel_size 25 \
        --revin 1 \
        --subtract_last 0 \
        --patch_size 56 \
        --patch_stride 24 \
        --patch_and_CI 1 \
        --d_seq 862 \
        --d_model 64 \
        --d_latent 16 \
        --fc_dropout 0.4 \
        --xlstm_h_num_block 1 \
        --slstm_h_at -1 \
        --xlstm_g_num_block 1 \
        --slstm_g_at -1 \
        --embed_hidden_dim 64 \
        --embed_hidden_layers_num 1 \
        --mlp_z_hidden_dim 32 \
        --mlp_z_hidden_layers_num 1 \
        --mlp_proj_down_hidden_dim 64 \
        --mlp_proj_down_hidden_layers_num 1 \
        --mlp_x_p_hidden_dim 64 \
        --mlp_x_p_hidden_layers_num 1 \
        --mlp_x_hidden_dim 64 \
        --mlp_x_hidden_layers_num 1 \
        --is_training 1 \
        --is_test 1 \
        --plot_result 1 \
        --train_epoches 512 \
        --batch_size 100 \
        --learning_rate 8e-4 \
        --weight_decay 1e-4 \
        --device 'cuda' \
        --figure_save_path ./figure/$data/$data'_'$prediction_length.pdf \
        --pre_train_wts_load_path  "" \
        --wts_load_path ./_wts/$data/$data'_'$prediction_length.ckpt \
        --wts_save_path ./_wts/$data/$data'_'$prediction_length.ckpt > ./train_info/$data/$data'_'$prediction_length.txt 2>&1
done