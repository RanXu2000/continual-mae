export PYTHONPATH=.
conda deactivate
conda activate vida
CUDA_VISIBLE_DEVICES=0 python cifar10c_vit_mae.py --cfg cfgs/cifar10/continual_mae.yaml --use_hog --hog_ratio 0.5