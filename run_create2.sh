sudo chmod 777 /dev/ttyUSB0
nohup python3 -u task_create2_visual_reacher.py --save_model --seed 1 --camera_id 0 --remote_ip "192.168.1.2" 1> train.out 2> train.err &
