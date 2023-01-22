import socket
import multiprocessing as mp
import queue
from relod.algo.comm import MODE
from relod.algo.rl_agent import BasePerformer, BaseLearner, BaseWrapper

class LocalWrapper(BaseWrapper):
    def __init__(self, max_samples_per_episode,
                       mode,
                       remote_ip='localhost', 
                       port=9876,
                       ):
        super().__init__()
        self._mode = mode
        print("Mode:", mode)
        if self._mode in [MODE.REMOTE_ONLY, MODE.REMOTE_LOCAL]:
            self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._cmd_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._cmd_sock.connect((remote_ip, port))
            print('Command socket connected!')

            self._data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._data_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._data_sock.connect((remote_ip, port+1))
            print('Data socket connected!')

            print('Sending mode to server...', end='')
            self.send_data(self._mode)
            print('done.')

            if self._mode == MODE.REMOTE_LOCAL:
                self._sample_queue = mp.Queue(3*max_samples_per_episode+100)
                self._policy_queue = mp.Queue(2)

                self._start_send_and_receive_event = mp.Event()
                self._send_started_event = mp.Event()
                self._receive_started_event = mp.Event()
                self._send_started_event.clear()
                self._receive_started_event.clear()

                # statistics
                self._sent_samples = mp.Value('L', 0, lock=False)
                self._received_policies = mp.Value('L', 0, lock=False)
                self._dropped_policies = mp.Value('L', 0, lock=False)
                self._applied_policies = 0

                self._send_p = mp.Process(target=self._send_sample_p)
                self._recv_p = mp.Process(target=self._receive_remote_policy_p)

                self._send_p.start()
                self._recv_p.start()

                # wait for processes to start
                self._send_started_event.wait()
                self._receive_started_event.wait()

        elif self._mode in [MODE.LOCAL_ONLY, MODE.EVALUATION]:
            pass
        else:
            raise NotImplementedError('init: {} mode is not supported'.format(self._mode))

    def _send_sample_p(self):
        print('Send process started.')
        self._send_started_event.set()
        while True:
            sample = self._sample_queue.get()
            self.send_data(sample)
            
            self._sent_samples.value += 1

    def _receive_remote_policy_p(self): # run in a child process
        print('Receive process started.')
        self._receive_started_event.set()
        while True:
            policy = self.recv_data()
            
            self._received_policies.value += 1
            try:
                self._policy_queue.put_nowait(policy)
            except queue.Full:
                self._dropped_policies.value += 1

    def init_performer(self, performer_class: BasePerformer, *args, **kwargs):
        if self._mode in [MODE.LOCAL_ONLY, MODE.REMOTE_LOCAL, MODE.EVALUATION]:
            self._performer = performer_class(*args, **kwargs)
        elif self._mode == MODE.REMOTE_ONLY:
            pass
        else:
            raise NotImplementedError('init_performer: {} mode is not supported'.format(self._mode))

    def init_learner(self, learner_class: BaseLearner, *args, **kwargs):
        if self._mode == MODE.LOCAL_ONLY:
            self._learner = learner_class(*args, **kwargs)
        elif self._mode in [MODE.REMOTE_LOCAL, MODE.REMOTE_ONLY, MODE.EVALUATION]:
            pass
        else:
            raise NotImplementedError('init_learner: {} mode is not supported'.format(self._mode))

    def send_init_ob(self, ob):
        if self._mode == MODE.REMOTE_ONLY:
            assert self.recv_cmd() == 'wait for new episode'
            self.send_data(ob)
        elif self._mode == MODE.REMOTE_LOCAL:
            assert self.recv_cmd() == 'wait for new episode'
            self._sample_queue.put_nowait(ob) # fatal error if sample queue is full
        elif self._mode in [MODE.LOCAL_ONLY, MODE.EVALUATION]:
            pass
        else:
            raise NotImplementedError('send_init_ob: {} mode is not supported'.format(self._mode))
    
    def push_sample(self, ob, action, reward, next_ob, done, *args, **kwargs):
        if self._mode == MODE.LOCAL_ONLY:
            self._learner.push_sample(ob, action, reward, next_ob, done, *args, **kwargs)
        elif self._mode == MODE.REMOTE_ONLY:
            self.send_data((reward, next_ob, done, *args, kwargs))
        elif self._mode == MODE.REMOTE_LOCAL:
            self._sample_queue.put_nowait((reward, next_ob, done, *args, kwargs)) # fatal error if sample queue is full
        elif self._mode == MODE.EVALUATION:
            pass
        else:
            raise NotImplementedError('push_sample: {} mode is not supported'.format(self._mode))

    def sample_action(self, ob, *args, **kwargs):
        if self._mode == MODE.REMOTE_ONLY:
            action = self.recv_data()
        elif self._mode in [MODE.REMOTE_LOCAL, MODE.LOCAL_ONLY, MODE.EVALUATION]:
            action = self._performer.sample_action(ob, *args, **kwargs)
            if self._mode == MODE.REMOTE_LOCAL:
                self._sample_queue.put_nowait(action)
        else:
            raise NotImplementedError('sample_action: {} mode is not supported'.format(self._mode))
        
        return action

    def apply_remote_policy(self, block=False):
        if self._mode == MODE.REMOTE_LOCAL:
            try:
                policy = self._policy_queue.get(block=block)
                self.performer.load_policy(policy)
                self._applied_policies += 1
                print('applied update:', self._applied_policies)
            except queue.Empty:
                pass

        elif self._mode in [MODE.REMOTE_ONLY, MODE.LOCAL_ONLY, MODE.EVALUATION]:
            pass
        else:
            raise NotImplementedError('recv_policy: {} mode is not supported'.format(self._mode))

    def update_policy(self, *args, **kwargs):
        if self._mode in [MODE.REMOTE_LOCAL, MODE.REMOTE_ONLY, MODE.EVALUATION]:
            if self._mode == MODE.REMOTE_LOCAL:
                self.apply_remote_policy()

            return None
        elif self._mode == MODE.LOCAL_ONLY:
            return self._learner.update_policy(*args, **kwargs)
        else:
            raise NotImplementedError('update_policy: {} mode is not supported'.format(self._mode))

    def save_policy_to_file(self, *args, **kwargs):
        if self._mode in [MODE.LOCAL_ONLY]:
            self._learner.save_policy_to_file(*args, **kwargs)
        elif self._mode in [MODE.REMOTE_ONLY, MODE.EVALUATION, MODE.REMOTE_LOCAL]:
            pass
        else:
            raise NotImplementedError('save_policy_to_file: {} mode is not supported'.format(self._mode))
    
    def load_policy_from_file(self, *args, **kwargs):
        if self._mode in [MODE.LOCAL_ONLY, MODE.EVALUATION]:
            self._performer.load_policy_from_file(*args, **kwargs)
        elif self._mode in [MODE.REMOTE_LOCAL, MODE.REMOTE_ONLY]:
            pass
        else:
            raise NotImplementedError('load_policy_to_file: {} mode is not supported'.format(self._mode))
    
    def save_buffer(self, *args, **kwargs):
        if self._mode == MODE.LOCAL_ONLY:
            self._learner.save_buffer(*args, **kwargs)
        elif self._mode in [MODE.REMOTE_ONLY, MODE.EVALUATION, MODE.REMOTE_LOCAL]:
            raise NotImplementedError('save_buffer: {} mode is not supported'.format(self._mode))
    

    def close(self, *args, **kwargs):
        if self._mode in [MODE.REMOTE_ONLY, MODE.REMOTE_LOCAL]:
            assert self.recv_cmd() == 'close'

            if self._mode == MODE.REMOTE_LOCAL:
                self._performer.close(*args, **kwargs)
                self._send_p.terminate()
                self._send_p.join()
                print('Send process finished')

                self._policy_queue.cancel_join_thread() # don't care the remaining policies 
                self._recv_p.terminate()
                self._recv_p.join()
                print('Receive process finished')
                print('Sent samples', self._sent_samples.value)
                print('Received policies', self._received_policies.value)
                print('Dropped policies', self._dropped_policies.value)
                print('Applied policies', self._applied_policies)
            
            self._cmd_sock.close()
            self._data_sock.close()
            
        elif self._mode in [MODE.LOCAL_ONLY, MODE.EVALUATION]:
            self._performer.close(*args, **kwargs)
            if self._mode == MODE.LOCAL_ONLY:
                self._learner.close(*args, **kwargs)
        else:
            raise NotImplementedError('close: {} mode is not supported'.format(self._mode))

        del self
