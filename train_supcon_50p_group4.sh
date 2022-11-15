CUDA_VISIBLE_DEVICES=1,2,3 python main_supcon.py --batch_size  256   --learning_rate 0.005  \
--temp 0.1 --cosine   --dataset path   --group_num "group4" \
--mean "(0.6958, 0.6816, 0.6524)"   --std "(0.3159, 0.3100, 0.3385)" \
--method SupCon  --data_folder /media0/chris/group4_resize_v2/train \
--wandb_id "newbornking999"
--save_freq 10  

