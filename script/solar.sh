if [ ! -d "./train_info" ]; then
    mkdir ./train_info
fi

if [ ! -d "./_wts" ]; then
    mkdir ./_wts
fi

root_path_name=./_dat/
random_seed=3407
look_back_length=336
data='Solar'


for prediction_length in 96 192 336 720
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
        --d_seq 137 \
        --d_model 64 \
        --d_latent 16 \
        --fc_dropout 0.2 \
        --xlstm_h_num_block 2 \
        --slstm_h_at 0 \
        --xlstm_g_num_block 1 \
        --slstm_g_at -1 \
        --embed_hidden_dim 64 \
        --embed_hidden_layers_num 1 \
        --mlp_z_hidden_dim 32 \
        --mlp_z_hidden_layers_num 2 \
        --mlp_proj_down_hidden_dim 64 \
        --mlp_proj_down_hidden_layers_num 1 \
        --mlp_x_p_hidden_dim 64 \
        --mlp_x_p_hidden_layers_num 1 \
        --mlp_x_hidden_dim 64 \
        --mlp_x_hidden_layers_num 2 \
        --is_training 0 \
        --is_test 0 \
        --plot_result 1 \
        --train_epoches 256 \
        --batch_size 22 \
        --learning_rate 3e-4 \
        --weight_decay 0 \
        --device 'cuda' \
        --figure_save_path ./figure/Solar/$data'_'$prediction_length.pdf \
        --pre_train_wts_load_path  "" \
        --wts_load_path ./_wts/Solar/$data'_'$prediction_length.ckpt \
        --wts_save_path ./_wts/$data'_'$prediction_length.ckpt > ./train_info/Solar/$data'_figure'$prediction_length.txt 2>&1
done