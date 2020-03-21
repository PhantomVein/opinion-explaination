export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

start=$(date +%Y-%m-%d_%H:%M:%S)
log_name=log_'$start'
nohup python -u driver/TrainTest.py --dataset hotel > $log_name 2>&1 &
# 最好将超参常用的写在程序参数中，用bash改
tail -f $log_name
