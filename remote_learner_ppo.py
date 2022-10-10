import torch
import argparse
from relod.algo.remote_wrapper import RemoteWrapper
from relod.algo.ppo_rad_agent import PPORADLearner, PPORADPerformer
from relod.algo.comm import MODE
from relod.logger import Logger
import time
import relod.utils as utils
import os

def parse_args():
    parser = argparse.ArgumentParser()

    # server
    parser.add_argument('--args_port', default=9630, type=int)
    # agent
    parser.add_argument('--port', default=9876, type=int)
    # misc
    parser.add_argument('--device', default='cuda:0', type=str)

    args = parser.parse_args()
    return args

def main():
    server_args = parse_args()

    agent = RemoteWrapper(port=server_args.port)
    args = agent.recv_data()
    agent.init_performer(PPORADPerformer, args)
    agent.init_learner(PPORADLearner, args, agent.performer)

    # sync initial weights with oboard
    agent.send_policy()

    utils.set_seed_everywhere(args.seed)

    utils.make_dir(args.work_dir)

    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    args.model_dir = model_dir

    if server_args.device is '':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        args.device = server_args.device

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, episode_step, done = 0, 0, 0, True
    episode_length_step = int(args.episode_length_time / args.dt)
    (image, propri) = agent.receive_init_ob()
    start_time = time.time()
    for step in range(args.env_steps):
        action, lprob = agent.sample_action((image, propri))
        
        (reward, (next_image, next_propri), done, lprob, kwargs) = agent.receive_sample_from_onboard()
        
        episode_reward += reward
        episode_step += 1

        agent.push_sample((image, propri), action, reward, (next_image, next_propri), done, lprob, **kwargs)

        if done or (episode_step == episode_length_step): # set time out here
            stat = agent.update_policy(done, next_image, next_propri)
            if agent.mode == MODE.REMOTE_LOCAL:
                if stat != None:
                    agent.send_cmd('new policy')
                    agent.send_policy()
                else:
                    agent.send_cmd('no policy')

            L.log('train/duration', time.time() - start_time, step)
            L.log('train/episode_reward', episode_reward, step)
            L.dump(step)
            (next_image, next_propri) = agent.receive_init_ob()
            episode_reward = 0
            episode_step = 0
            episode += 1
            L.log('train/episode', episode, step)
            start_time = time.time()
            
        
        (image, propri) = (next_image, next_propri)

        if args.save_model and (step+1) % args.save_model_freq == 0:
            agent.save_policy_to_file(step)

    agent.save_policy_to_file(step)
    agent.close()
    print('Train finished')

if __name__ == '__main__':
    main()
