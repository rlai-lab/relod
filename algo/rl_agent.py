class BaseLearner:
    def get_policy(self, *args, **kwargs):
        raise NotImplementedError()

    def update_policy(self, *args, **kwargs):
        raise NotImplementedError()

    def push_sample(self, ob, action, reward, next_ob, done, *args, **kwargs):
        raise NotImplementedError()

    def close(self, *args, **kwargs):
        raise NotImplementedError()


class BasePerformer:
    def init_policy(self, *args, **kwargs):
        raise NotImplementedError()

    def apply_remote_policy(self, policy, *args, **kwargs):
        raise NotImplementedError()

    def sample_action(self, ob, *args, **kwargs):
        raise NotImplementedError()

    def close(self, *args, **kwargs):
        raise NotImplementedError()
