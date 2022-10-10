from relod.algo.comm import MODE, send_message, recv_message

class BaseWrapper:
    def __init__(self) -> None:
        self._performer = None
        self._learner = None
        self._mode = MODE.REMOTE_ONLY
        self._data_sock = None
        self._cmd_sock = None
         
    def init_performer(self, *args, **kwargs):
        raise NotImplementedError()

    def init_learner(self, *args, **kwargs):
        raise NotImplementedError()

    def save_policy_to_file(self, *args, **kwargs):
        raise NotImplementedError()
    
    def load_policy_from_file(self, *args, **kwargs):
        raise NotImplementedError()

    def sample_action(self, ob, *args, **kwargs):
        raise NotImplementedError()

    def push_sample(self, *args, **kwargs):
        raise NotImplementedError()

    def update_policy(self, *args, **kwargs):
        raise NotImplementedError()

    def close(self, *args, **kwargs):
        raise NotImplementedError()

    def send_cmd(self, cmd):
        if self._mode in [MODE.REMOTE_LOCAL, MODE.REMOTE_ONLY]:
            send_message(cmd, self._cmd_sock)
        elif self._mode in [MODE.LOCAL_ONLY, MODE.EVALUATION]:
            pass
        else:
            raise NotImplementedError('send_cmd: {} mode is not supported'.format(self._mode))

    def recv_cmd(self):
        if self._mode in [MODE.REMOTE_LOCAL, MODE.REMOTE_ONLY]:
            return recv_message(self._cmd_sock)
        elif self._mode in [MODE.LOCAL_ONLY, MODE.EVALUATION]:
            pass
        else:
            raise NotImplementedError('recv_cmd: {} mode is not supported'.format(self._mode))

    def send_data(self, msg):
        if self._mode in [MODE.REMOTE_LOCAL, MODE.REMOTE_ONLY]:
            send_message(msg, self._data_sock)
        elif self._mode in [MODE.LOCAL_ONLY, MODE.EVALUATION]:
            pass
        else:
            raise NotImplementedError('send_data: {} mode is not supported'.format(self._mode))

    def recv_data(self):
        if self._mode in [MODE.REMOTE_LOCAL, MODE.REMOTE_ONLY]:
            return recv_message(self._data_sock)
        elif self._mode in [MODE.LOCAL_ONLY, MODE.EVALUATION]:
            pass
        else:
            raise NotImplementedError('recv_data: {} mode is not supported'.format(self._mode))

    @property
    def performer(self):
        return self._performer

    @property
    def learner(self):
        return self._learner

    @property
    def mode(self):
        return self._mode

class BaseLearner:
    def get_policy(self, *args, **kwargs):
        raise NotImplementedError()

    def update_policy(self, *args, **kwargs):
        raise NotImplementedError()

    def push_sample(self, ob, action, reward, next_ob, done, *args, **kwargs):
        raise NotImplementedError()

    def save_policy_to_file(self, *args, **kwargs):
        raise NotImplementedError()

    def load_policy_from_file(self, *args, **kwargs):
        raise NotImplementedError()

    def close(self, *args, **kwargs):
        raise NotImplementedError()

class BasePerformer:
    def load_policy(self, policy, *args, **kwargs):
        raise NotImplementedError()

    def sample_action(self, ob, *args, **kwargs):
        raise NotImplementedError()

    def load_policy_from_file(self, *args, **kwargs):
        raise NotImplementedError()

    def close(self, *args, **kwargs):
        raise NotImplementedError()
