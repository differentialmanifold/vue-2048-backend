import redis
import numpy as np

host = '127.0.0.1'
port = 6379
password = None


class MyRedis:
    def __init__(self, scope):
        self.my_redis = redis.StrictRedis(host=host, port=port, db=0, password=password)
        self.scope = scope
        self.store_batch = 100000

    def store_td(self, saved_obj):
        my_dicts = saved_obj['q']
        step = saved_obj['step']

        self.my_redis.set(self.scope + ':step', step)

        i = 0
        tmp = dict()
        for item in my_dicts.items():
            tmp[item[0]] = '_'.join(['{:0.3f}'.format(value) for value in item[1]])
            i = i + 1
            if i % self.store_batch == 0:
                self.my_redis.hmset(self.scope + ':q', tmp)
                tmp = dict()
        if len(tmp) > 0:
            self.my_redis.hmset(self.scope + ':q', tmp)

    def restore_td(self):
        has_step = self.my_redis.exists(self.scope + ':step')
        has_q_dicts = self.my_redis.exists(self.scope + ':q')
        if not has_step or not has_q_dicts:
            return None

        restore_dicts = dict()
        restore_dicts['step'] = int(self.my_redis.get(self.scope + ':step').decode('utf-8'))
        q_dicts = self.my_redis.hgetall(self.scope + ':q')

        temp_dicts = dict()
        for pair in q_dicts.items():
            temp_dicts[pair[0].decode('utf-8')] = np.array([float(item) for item in pair[1].decode('utf-8').split('_')])
        restore_dicts['q'] = temp_dicts

        return restore_dicts

    def hlen_q(self):
        name = self.scope + ':q'
        value = self.my_redis.hlen(name)

        return value

    def hget_q(self, key):
        name = self.scope + ':q'
        value = self.my_redis.hget(name, key)

        if value is None:
            return np.zeros(4)
        return np.array([float(item) for item in value.decode('utf-8').split('_')])

    def hset_q(self, key, value):
        name = self.scope + ':q'
        self.my_redis.hset(name, key, '_'.join(['{:0.3f}'.format(value) for value in value]))

    def get_step(self):
        value = self.my_redis.get(self.scope + ':step').decode('utf-8')

        if value is None:
            return -1

        return int(value)

    def set_step(self, value):
        self.my_redis.set(self.scope + ':step', value)
