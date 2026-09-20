lr怎么设定呢？
full batch step_num = 5000 lr = 1 没问题
mini-batch epoch = 5000 num_samples = 500 batch_size = 32  一个epoch 更新500/32 = 15.612 也就是16次 lr = 1 训练直接崩掉
