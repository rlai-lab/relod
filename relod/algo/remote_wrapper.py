from relod.algo.comm import MODE
from relod.algo.rl_agent import BaseLearner, BasePerformer, BaseWrapper
import socket

class RemoteWrapper(BaseWrapper):
    def __init__(self, port=9876):
        super().__init__()
        server_cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_cmd_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server_data_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        server_cmd_sock.bind(('', port))
        server_data_sock.bind(('', port+1))

        server_cmd_sock.listen(1)
        server_data_sock.listen(1)

        print('Command socket created, listening...')
        print('Data socket created, listening...')

        (self._cmd_sock, address) = server_cmd_sock.accept()
        (self._data_sock, address) = server_data_sock.accept()
        print('Command and Data sockets are connected, ip:', address)

        self._mode = self.recv_data()
        print("Mode:", self._mode)

    def init_performer(self, performer_class: BasePerformer, *args, **kwargs):
        if self._mode == MODE.REMOTE_ONLY:
            self._performer = performer_class(*args, **kwargs)
        elif self._mode == MODE.REMOTE_LOCAL:
            pass
        else:
            raise NotImplementedError('init_performer: {} mode is not supported'.format(self._mode))

    def init_learner(self, learner_class: BaseLearner, *args, **kwargs):
        if self._mode in [MODE.REMOTE_ONLY, MODE.REMOTE_LOCAL]:
            self._learner = learner_class(*args, **kwargs)
        else:
            raise NotImplementedError('init_learner: {} mode is not supported'.format(self._mode))

    def receive_init_ob(self):
        if self._mode in [MODE.REMOTE_LOCAL, MODE.REMOTE_ONLY]:
            self.send_cmd('wait for new episode')
            return self.recv_data()
        else:
            raise NotImplementedError('receive_init_ob: {} mode is not supported'.format(self._mode))

    def sample_action(self, ob, *args, **kwargs):
        if self._mode == MODE.REMOTE_ONLY:
            action = self._performer.sample_action(ob, *args, **kwargs)
            self.send_data(action)
        elif self._mode == MODE.REMOTE_LOCAL:
            action = self.recv_data()
        else:
            raise NotImplementedError('sample_action: {} mode is not supported'.format(self._mode))

        return action

    def receive_sample_from_onboard(self):
        if self._mode in [MODE.REMOTE_ONLY, MODE.REMOTE_LOCAL]:
            return self.recv_data()
        else: 
            raise NotImplementedError('receive_sample: {} mode is not supported'.format(self._mode))

    def push_sample(self, ob, action, reward, next_ob, done, *args, **kwargs):
        if self._mode in [MODE.REMOTE_ONLY, MODE.REMOTE_LOCAL]:
            self._learner.push_sample(ob, action, reward, next_ob, done, *args, **kwargs)
        else:
            raise NotImplementedError('push_sample: {} mode is not supported'.format(self._mode))
            
    def send_policy(self):
        if self._mode == MODE.REMOTE_LOCAL:
            self.send_data(self.learner.get_policy())
        elif self._mode == MODE.REMOTE_ONLY:
            pass
        else:
            raise NotImplementedError('send_policy: {} mode is not supported'.format(self._mode))

    def update_policy(self, *args, **kwargs):
        if self._mode in [MODE.REMOTE_ONLY, MODE.REMOTE_LOCAL]:
            return self._learner.update_policy(*args, **kwargs)
        else:
            raise NotImplementedError('update_policy: {} mode is not supported'.format(self._mode))

    def save_policy_to_file(self, *args, **kwargs):
        if self._mode in [MODE.REMOTE_ONLY, MODE.REMOTE_LOCAL]:
            self._learner.save_policy_to_file(*args, **kwargs)
        else:
            raise NotImplementedError('save_policy_to_file: {} mode is not supported'.format(self._mode))
    
    def load_policy_from_file(self, *args, **kwargs):
        if self._mode in [MODE.REMOTE_ONLY, MODE.REMOTE_LOCAL]:
            self._learner.load_policy_from_file(*args, **kwargs)
        else:
            raise NotImplementedError('load_policy_to_file: {} mode is not supported'.format(self._mode))
    
    def close(self, *args, **kwargs):
        if self._mode in [MODE.REMOTE_ONLY, MODE.REMOTE_LOCAL]:
            self.send_cmd('close')
            if self._mode == MODE.REMOTE_ONLY:
                self._performer.close(*args, **kwargs)

            self._learner.close(*args, **kwargs)  
            self._cmd_sock.close()
            self._data_sock.close()
        else:
            raise NotImplementedError('close: {} mode is not supported'.format(self._mode))

        del self
