if [ ! -d "./train_info/Electricity" ]; then
    mkdir -p ./train_info/Electricity
fi

if [ ! -d "./_wts/Electricity" ]; then
    mkdir -p ./_wts/Electricity
fi

if [ ! -d "./figure/Electricity" ]; then
    mkdir -p ./figure/Electricity
fi

root_path_name=./_dat/
random_seed=3407
data='Electricity'
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
    --d_seq 321 \
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
    --is_training 1 \
    --is_test 1 \
    --plot_result 1 \
    --train_epoches 256 \
    --batch_size 40 \
    --learning_rate 3e-4 \
    --weight_decay 0 \
    --device 'cuda' \
    --figure_save_path ./figure/$data/$data'_96'.pdf \
    --pre_train_wts_load_path  "" \
    --wts_load_path ./_wts/$data/$data'_96'.ckpt \
    --wts_save_path ./_wts/$data/$data'_96'.ckpt > ./train_info/$data/$data'_96'.txt 2>&1


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
    --d_seq 321 \
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
    --is_training 1 \
    --is_test 1 \
    --plot_result 1 \
    --train_epoches 256 \
    --batch_size 25 \
    --learning_rate 3e-4 \
    --weight_decay 0 \
    --device 'cuda' \
    --figure_save_path ./figure/$data/$data'_192'.pdf \
    --pre_train_wts_load_path  "" \
    --wts_load_path ./_wts/$data/$data'_192'.ckpt \
    --wts_save_path ./_wts/$data/$data'_192'.ckpt > ./train_info/$data/$data'_192'.txt 2>&1


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
    --d_seq 321 \
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
    --is_training 1 \
    --is_test 1 \
    --plot_result 1 \
    --train_epoches 256 \
    --batch_size 15 \
    --learning_rate 3e-4 \
    --weight_decay 0 \
    --device 'cuda' \
    --figure_save_path ./figure/$data/$data'_336'.pdf \
    --pre_train_wts_load_path  "" \
    --wts_load_path ./_wts/$data/$data'_336'.ckpt \
    --wts_save_path ./_wts/$data/$data'_336'.ckpt > ./train_info/$data/$data'_336'.txt 2>&1


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
    --d_seq 321 \
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
    --is_training 1 \
    --is_test 1 \
    --plot_result 1 \
    --train_epoches 256 \
    --batch_size 10 \
    --learning_rate 3e-4 \
    --weight_decay 0 \
    --device 'cuda' \
    --figure_save_path ./figure/$data/$data'_720'.pdf \
    --pre_train_wts_load_path  "" \
    --wts_load_path ./_wts/$data/$data'_720'.ckpt \
    --wts_save_path ./_wts/$data/$data'_720'.ckpt > ./train_info/$data/$data'_720'.txt 2>&1