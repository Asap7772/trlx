which_gpu="0,1,2,3,4"
num_gpu=5
port_num=29500

command="OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=$which_gpu accelerate launch \
    --num_processes $num_gpu --main_process_port $port_num \
    --config_file configs/accelerate/zero2-bf16.yaml examples/ppo_sentiments.py"

echo $command
eval $command