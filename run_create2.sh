sudo chmod 777 /dev/ttyUSB0
nohup python3 -u task_create2_visual_reacher.py --run_type init_policy_test --sync_mode --work_dir init_policy_test --replay_buffer_capacity 1000 1> train.out 2> train.err &
