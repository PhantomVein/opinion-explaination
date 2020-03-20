export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

log_name=log_'$starttime'
nohup python -u driver/TrainTest.py --config_file config > $log_name 2>&1 & 
# 最好将超参常用的写在程序参数中，用bash改
tail -f $log_name
