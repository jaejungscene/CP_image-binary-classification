python train.py --model efficientnet_b6 --lr 1e-5 --eta_max 1e-4 --epoch 50 --batch_size 16 --wandb 1 --cuda 2,3,4,5 --optimizer adamw --scheduler cosinerestarts #--label_smooth 0.1
