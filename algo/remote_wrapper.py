from algo.comm import MODE, send_message, recv_message
import socket

class RemoteWrapper:
    def __init__(self, port=9876, performer=None, learner=None):

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

        (cmd_sock, address) = server_cmd_sock.accept()
        (data_sock, address) = server_data_sock.accept()
        print('Command and Data sockets are connected.')

        mode = recv_message(data_sock)
        print("Mode:", mode)
        self._cmd_sock = cmd_sock
        self._data_sock = data_sock
        self._mode = mode
        self._performer = performer
        self._learner = learner

        if mode == MODE.REMOTE_ONLY:
            assert performer != None and learner != None
        elif mode == MODE.ONBOARD_REMOTE:
            assert  performer == None and learner != None
        else:
            raise NotImplementedError()

        if mode == MODE.ONBOARD_REMOTE:
            send_message(self._learner.get_policy(), data_sock)

    def receive_init_ob(self):
        send_message('wait for new episode', self._cmd_sock)
        return recv_message(self._data_sock)

    def sample_action(self, ob, *args, **kwargs):
        if self._mode == MODE.REMOTE_ONLY:
            action = self._performer.sample_action(ob, *args, **kwargs)
            send_message(action, self._data_sock)
        elif self._mode == MODE.ONBOARD_REMOTE:
            action = recv_message(self._data_sock)
        else:
            raise NotImplementedError('sample_action: {} mode is not supported'.format(self._mode))

        return action

    def receive_sample(self):
        if self._mode in [MODE.REMOTE_ONLY, MODE.ONBOARD_REMOTE]:
            return recv_message(self._data_sock)
        else: 
            raise NotImplementedError('receive_sample: {} mode is not supported'.format(self._mode))

    def update_policy(self, *args, **kwargs):
        val = self._learner.update_policy(*args, **kwargs)
        if self._mode == MODE.REMOTE_ONLY:
            send_message(val, self._data_sock)
        elif self._mode == MODE.ONBOARD_REMOTE:
            if val is not None:
                policy = self._learner.get_policy()
                send_message((val, policy), self._data_sock)
        else:
            raise NotImplementedError('update_policy: {} mode is not supported'.format(self._mode))
        
        return val

    def close(self, *args, **kwargs):
        self._learner.close(*args, **kwargs)
        if self._mode == MODE.REMOTE_ONLY:
            self._performer.close(*args, **kwargs)
        else:
            raise NotImplementedError()

        send_message('close', self._cmd_sock)
        if self._mode == MODE.ONBOARD_REMOTE:
            assert recv_message(self._cmd_sock) == 'close sockets' # this is needed to prevent exception on receive_p 
        
        self._cmd_sock.close()
        self._data_sock.close()

        del self
