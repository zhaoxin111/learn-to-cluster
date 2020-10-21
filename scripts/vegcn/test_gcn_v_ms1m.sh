config=vegcn/configs/cfg_test_gcnv_ms1m.py
# load_from=data/pretrained_models/pretrained_gcn_v_ms1m.pth
load_from=data/work_dir/cfg_train_gcnv_ms1m/latest.pth

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=.

python vegcn/main.py \
    --config $config \
    --phase 'test' \
    --load_from $load_from \
    --save_output \
    --eval_interim \
    --force
