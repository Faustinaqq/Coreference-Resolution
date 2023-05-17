# LSTM + Attention
CUDA_VISIBLE_DEVICES=2 python main.py --sample_size 100 --lr 1e-03 --epochs 500 \
--eval_epoch 1 --log_path './log/log1.txt' --model 'lstm' --attention 1

#LSTM
# CUDA_VISIBLE_DEVICES=2 python main.py --sample_size 100 --lr 1e-03 --epochs 500 \
# --eval_epoch 1 --log_path './log/log2.txt' --model 'lstm' --attention 0 \
# --pretrained_vector_path '/home/fqq/data/sgns.merge.word.bz2'

# #RNN
# CUDA_VISIBLE_DEVICES=2 python main.py --sample_size 100 --lr 1e-03 --epochs 500 \
# --eval_epoch 1 --log_path './log/log3.txt' --model 'rnn' --attention 0 \
# --pretrained_vector_path '/home/fqq/data/sgns.merge.word.bz2'