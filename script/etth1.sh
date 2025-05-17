if [ ! -d "./train_info/ETTh1" ]; then
    mkdir ./train_info/ETTh1
fi

if [ ! -d "./_wts/ETTh1" ]; then
    mkdir ./_wts/ETTh1
fi

if [ ! -d "./figure/ETTh1" ]; then
    mkdir ./figure/ETTh1
fi

root_path_name=./_dat/
random_seed=3407
data='ETTh1'
look_back_length=336

python -u main.py \
    --random_seed $random_seed \
    --data $data \
    --root_path $root_path_name \
    --features 'M' \
    --look_back_length $look_back_length \
    --label_len $look_back_length \
    --prediction_length 96 \
    --decomposition 1 \
    --kernel_size 25 \
    --revin 1 \
    --subtract_last 0 \
    --patch_size 56 \
    --patch_stride 24 \
    --patch_and_CI 1 \
    --d_seq 7 \
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
    --batch_size 450 \
    --learning_rate 6e-4 \
    --weight_decay 1e-4 \
    --device 'cuda' \
    --figure_save_path ./figure/ETTh1/$data'_96'.pdf \
    --pre_train_wts_load_path  "" \
    --wts_load_path ./_wts/ETTh1/$data'_96'.ckpt \
    --wts_save_path ./_wts/ETTh1/$data'_96'.ckpt > ./train_info/ETTh1/$data'_96'.txt 2>&1


python -u main.py \
    --random_seed $random_seed \
    --data $data \
    --root_path $root_path_name \
    --features 'M' \
    --look_back_length $look_back_length \
    --label_len $look_back_length \
    --prediction_length 192 \
    --decomposition 1 \
    --kernel_size 25 \
    --revin 1 \
    --subtract_last 0 \
    --patch_size 56 \
    --patch_stride 24 \
    --patch_and_CI 1 \
    --d_seq 7 \
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
    --batch_size 450 \
    --learning_rate 6e-4 \
    --weight_decay 1e-3 \
    --device 'cuda' \
    --figure_save_path ./figure/ETTh1/$data'_192'.pdf \
    --pre_train_wts_load_path  "" \
    --wts_load_path ./_wts/ETTh1/$data'_192'.ckpt \
    --wts_save_path ./_wts/ETTh1/$data'_192'.ckpt > ./train_info/ETTh1/$data'_192'.txt 2>&1


python -u main.py \
    --random_seed $random_seed \
    --data $data \
    --root_path $root_path_name \
    --features 'M' \
    --look_back_length $look_back_length \
    --label_len $look_back_length \
    --prediction_length 336 \
    --decomposition 1 \
    --kernel_size 25 \
    --revin 1 \
    --subtract_last 0 \
    --patch_size 56 \
    --patch_stride 24 \
    --patch_and_CI 1 \
    --d_seq 7 \
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
    --train_epoches 512 \
    --batch_size 450 \
    --learning_rate 6e-4 \
    --weight_decay 1e-4 \
    --device 'cuda' \
    --figure_save_path ./figure/ETTh1/$data'_336'.pdf \
    --pre_train_wts_load_path  "" \
    --wts_load_path ./_wts/ETTh1/$data'_336'.ckpt \
    --wts_save_path ./_wts/ETTh1/$data'_336'.ckpt > ./train_info/ETTh1/$data'_336'.txt 2>&1


python -u main.py \
    --random_seed $random_seed \
    --data $data \
    --root_path $root_path_name \
    --features 'M' \
    --look_back_length $look_back_length \
    --label_len $look_back_length \
    --prediction_length 720 \
    --decomposition 1 \
    --kernel_size 25 \
    --revin 1 \
    --subtract_last 0 \
    --patch_size 56 \
    --patch_stride 24 \
    --patch_and_CI 1 \
    --d_seq 7 \
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
    --batch_size 450 \
    --learning_rate 6e-4 \
    --weight_decay 1e-2 \
    --device 'cuda' \
    --figure_save_path ./figure/ETTh1/$data'_720'.pdf \
    --pre_train_wts_load_path  "" \
    --wts_load_path ./_wts/ETTh1/$data'_720'.ckpt \
    --wts_save_path ./_wts/ETTh1/$data'_720'.ckpt > ./train_info/ETTh1/$data'_720'.txt 2>&1