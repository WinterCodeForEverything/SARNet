CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12363 \
    ../train.py --dataset NuScenes --semantic_classes_num 14 --sample_voxel_size 0.3 \
    --sample_point_num 8000 --boundingbox_diagonal 80;
CUDA_VISIBLE_DEVICES=1 python ../test.py  --dataset NuScenes --semantic_classes_num 14 --sample_voxel_size 0.3 \
    --sample_point_num 8000 --boundingbox_diagonal 80;