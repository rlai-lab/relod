import socket
import multiprocessing as mp
import queue
from algo.comm import MODE, send_message, recv_message

class OnboardWrapper:
    def __init__(self, max_samples_per_episode,
                       mode,
                       remote_ip='localhost', 
                       port=9876,
                       performer=None,
                       learner=None):

        if mode == MODE.LOCAL_ONLY:
            assert performer != None and learner != None
        elif mode == MODE.ONBOARD_REMOTE or MODE.REMOTE_ONLY:
            assert  performer != None and learner == None
        else:
            raise NotImplementedError()

        self._mode = mode
        self._performer = performer
        self._learner = learner
        if self._mode in [MODE.REMOTE_ONLY, MODE.ONBOARD_REMOTE]:
            self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._cmd_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._cmd_sock.connect((remote_ip, port))
            print('Command socket connected!')

            self._data_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._data_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._data_sock.connect((remote_ip, port+1))
            print('Data socket connected!')

            print('Sending mode to server...', end='')
            send_message(self._mode, self._data_sock)
            print('done.')

        if self._mode == MODE.ONBOARD_REMOTE:
            self._sample_queue = mp.Queue(3*max_samples_per_episode+100)
            self._policy_queue = mp.Queue(2)

            self._start_send_and_receive_event = mp.Event()
            self._send_started_event = mp.Event()
            self._receive_started_event = mp.Event()
            self._start_send_and_receive_event.clear()
            self._send_started_event.clear()
            self._receive_started_event.clear()

            # statistics
            self._sent_samples = mp.Value('L', 0, lock=False)
            self._dropped_samples = mp.Value('L', 0, lock=False)
            self._received_policies = mp.Value('L', 0, lock=False)
            self._dropped_policies = mp.Value('L', 0, lock=False)
            self._applied_policies = mp.Value('L', 0, lock=False)

            self._send_p = mp.Process(target=self._send_sample)
            self._recv_p = mp.Process(target=self._receive_remote_policy)

            self._send_p.start()
            self._recv_p.start()

            self._performer.init_policy()

            print('Receiving policy from server...', end='')
            policy = recv_message(self._data_sock)
            self._performer.apply_remote_policy(policy)
            print('done')

            self._start_send_and_receive_event.set()
            self._send_started_event.wait()
            self._receive_started_event.wait()

        if self._mode == MODE.LOCAL_ONLY:
            pass

    def _send_sample(self):
        self._start_send_and_receive_event.wait() # wait for main process to finish handshake
        print('Send process started.')
        self._send_started_event.set()
        while True:
            sample = self._sample_queue.get()
            send_message(sample, self._data_sock)
            
            if len(sample) == 3:
                self._sent_samples.value += 1

    def _receive_remote_policy(self): # run in a child process
        self._start_send_and_receive_event.wait() # wait for main process to finish handshake
        print('Receive process started.')
        self._receive_started_event.set()
        while True:
            policy = recv_message(self._data_sock)
            
            self._received_policies.value += 1
            try:
                self._policy_queue.put_nowait(policy)
            except queue.Full:
                self._dropped_policies.value += 1

    def send_init_ob(self, ob):
        if self._mode == MODE.REMOTE_ONLY:
            assert recv_message(self._cmd_sock) == 'wait for new episode'
            send_message(ob, self._data_sock)
        elif self._mode == MODE.ONBOARD_REMOTE:
            assert recv_message(self._cmd_sock) == 'wait for new episode'
            self._sample_queue.put_nowait(ob) # fatal error if sample queue is full
        elif self._mode == MODE.LOCAL_ONLY:
            pass
        else:
            raise NotImplementedError('send_init_ob: {} mode is not supported'.format(self._mode))
    
    def push_sample(self, ob, action, reward, next_ob, done, *args, **kwargs):
        if self._mode == MODE.LOCAL_ONLY:
            self._learner.push_sample(ob, action, reward, next_ob, done)
        elif self._mode == MODE.REMOTE_ONLY:
            send_message((reward, next_ob, done), self._data_sock)
        elif self._mode == MODE.ONBOARD_REMOTE:
            self._sample_queue.put_nowait((reward, next_ob, done)) # fatal error if sample queue is full
        else:
            raise NotImplementedError('push_sample: {} mode is not supported'.format(self._mode))

    def sample_action(self, ob, *args, **kwargs):
        if self._mode == MODE.REMOTE_ONLY:
            action = recv_message(self._data_sock)
        elif self._mode == MODE.ONBOARD_REMOTE:
            action = self._performer.sample_action(ob, *args, **kwargs)
            self._sample_queue.put_nowait(action)
        elif self._mode == MODE.LOCAL_ONLY:
            action = self._performer.sample_action(ob, *args, **kwargs)
        else:
            raise NotImplementedError('sample_action: {} mode is not supported'.format(self._mode))
        
        return action

    def update_policy(self, *args, **kwargs):
        if self._mode == MODE.ONBOARD_REMOTE:
            try:
                (val, policy) = self._policy_queue.get_nowait()
                self._performer.apply_remote_policy(policy)
                self._applied_policies.value += 1
                return val
            except queue.Empty:
                return None
        elif self._mode == MODE.LOCAL_ONLY:
            val = self._learner.update_policy(*args, **kwargs)
            return val
        elif self._mode == MODE.REMOTE_ONLY:
            return recv_message(self._data_sock)
        else:
            raise NotImplementedError('update_policy: {} mode is not supported'.format(self._mode))

    def close(self, *args, **kwargs):
        if self._mode == MODE.ONBOARD_REMOTE:
            self._performer.close(*args, **kwargs)
            assert recv_message(self._cmd_sock) == 'close'

            self._send_p.terminate()
            self._send_p.join()
            print('Send process finished')

            self._recv_p.terminate()
            self._recv_p.join()
            print('Receive process finished')
            
            send_message('close sockets', self._cmd_sock)
            self._cmd_sock.close()
            self._data_sock.close()

            print('Sent samples', self._sent_samples.value)
            print('Dropped samples', self._dropped_samples.value)
            print('Received policies', self._received_policies.value)
            print('Dropped policies', self._dropped_policies.value)
            print('Applied policies', self._applied_policies.value)
            
        elif self._mode == MODE.REMOTE_ONLY:
            self._performer.close(*args, **kwargs)

            assert recv_message(self._cmd_sock) == 'close'
            self._cmd_sock.close()
            self._data_sock.close()
        elif self._mode == MODE.LOCAL_ONLY:
            self._performer.close(*args, **kwargs)
            self._learner.close(*args, **kwargs)
        else:
            raise NotImplementedError('close: {} mode is not supported'.format(self._mode))

        del self

    @property
    def mode(self):
        return self._mode
