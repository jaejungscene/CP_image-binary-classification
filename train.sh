python train.py --model efficientnet_b6 --lr 1e-4 --epoch 50 --batch_size 32 --wandb 1 --cuda 0,1,6,7,8 --optimizer sgd  --scheduler cosinerestarts