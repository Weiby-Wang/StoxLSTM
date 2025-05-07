if [ ! -d "./train_info" ]; then
    mkdir ./train_info
fi

if [ ! -d "./_wts" ]; then
    mkdir ./_wts
fi

root_path_name=./_dat/
random_seed=3407
look_back_length=96

python -u main.py \
    --random_seed $random_seed \
    --data 'PEMS03' \
    --root_path $root_path_name \
    --features 'M' \
    --look_back_length $look_back_length \
    --label_len $look_back_length \
    --prediction_length 12 \
    --decomposition 1 \
    --kernel_size 25 \
    --revin 1 \
    --subtract_last 0 \
    --patch_size 56 \
    --patch_stride 24 \
    --patch_and_CI 1 \
    --d_seq 358 \
    --d_model 64 \
    --d_latent 16 \
    --fc_dropout 0.2 \
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
    --train_epoches 128 \
    --batch_size 80 \
    --learning_rate 3e-4 \
    --weight_decay 0 \
    --device 'cuda' \
    --figure_save_path ./figure/PEMS03_12.png \
    --pre_train_wts_load_path  "" \
    --wts_load_path ./_wts/PEMS03_12.ckpt \
    --wts_save_path ./_wts/PEMS03_12.ckpt > ./train_info/PEMS03_12.txt 2>&1


python -u main.py \
    --random_seed $random_seed \
    --data 'PEMS04' \
    --root_path $root_path_name \
    --features 'M' \
    --look_back_length $look_back_length \
    --label_len $look_back_length \
    --prediction_length 12 \
    --decomposition 1 \
    --kernel_size 25 \
    --revin 1 \
    --subtract_last 0 \
    --patch_size 56 \
    --patch_stride 24 \
    --patch_and_CI 1 \
    --d_seq 307 \
    --d_model 64 \
    --d_latent 16 \
    --fc_dropout 0.2 \
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
    --train_epoches 128 \
    --batch_size 80 \
    --learning_rate 3e-4 \
    --weight_decay 0 \
    --device 'cuda' \
    --figure_save_path ./figure/PEMS04_12.png \
    --pre_train_wts_load_path  "" \
    --wts_load_path ./_wts/PEMS04_12.ckpt \
    --wts_save_path ./_wts/PEMS04_12.ckpt > ./train_info/PEMS04_12.txt 2>&1


python -u main.py \
    --random_seed $random_seed \
    --data 'PEMS07' \
    --root_path $root_path_name \
    --features 'M' \
    --look_back_length $look_back_length \
    --label_len $look_back_length \
    --prediction_length 12 \
    --decomposition 1 \
    --kernel_size 25 \
    --revin 1 \
    --subtract_last 0 \
    --patch_size 56 \
    --patch_stride 24 \
    --patch_and_CI 1 \
    --d_seq 883 \
    --d_model 64 \
    --d_latent 16 \
    --fc_dropout 0.2 \
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
    --train_epoches 128 \
    --batch_size 35 \
    --learning_rate 3e-4 \
    --weight_decay 0 \
    --device 'cuda' \
    --figure_save_path ./figure/PEMS07_12.png \
    --pre_train_wts_load_path  "" \
    --wts_load_path ./_wts/PEMS07_12.ckpt \
    --wts_save_path ./_wts/PEMS07_12.ckpt > ./train_info/PEMS07_12.txt 2>&1


python -u main.py \
    --random_seed $random_seed \
    --data 'PEMS08' \
    --root_path $root_path_name \
    --features 'M' \
    --look_back_length $look_back_length \
    --label_len $look_back_length \
    --prediction_length 12 \
    --decomposition 1 \
    --kernel_size 25 \
    --revin 1 \
    --subtract_last 0 \
    --patch_size 56 \
    --patch_stride 24 \
    --patch_and_CI 1 \
    --d_seq 170 \
    --d_model 64 \
    --d_latent 16 \
    --fc_dropout 0.2 \
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
    --train_epoches  128 \
    --batch_size 160 \
    --learning_rate 3e-4 \
    --weight_decay 0 \
    --device 'cuda' \
    --figure_save_path ./figure/PEMS08_12.png \
    --pre_train_wts_load_path  "" \
    --wts_load_path ./_wts/PEMS08_12.ckpt \
    --wts_save_path ./_wts/PEMS08_12.ckpt > ./train_info/PEMS08_12.txt 2>&1