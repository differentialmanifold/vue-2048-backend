import pickle
from collections import defaultdict
import numpy as np
import itertools
import time
from collections import namedtuple
import random
import math

import scipy.io

import tensorflow as tf

from sklearn.externals import joblib

import redis


def actions():
    return np.zeros(4)


def store_redis(my_dicts, step, my_redis, scope):
    # with open('temp' + '.pickle', 'wb') as f:
    #     pickle.dump(saved_obj, f, pickle.HIGHEST_PROTOCOL)
    my_redis.set(scope + ':step', step)

    gase = 100000
    i = 0
    tmp = dict()
    for item in my_dicts.items():
        tmp[item[0]] = item[1]
        i = i + 1
        if i % gase == 0:
            print('iterate is {}'.format(i))
            my_redis.hmset(scope + ':q', tmp)
            tmp = dict()
    if len(tmp) > 0:
        my_redis.hmset(scope + ':q', tmp)


def restore_redis(my_redis, scope):
    # with open('temp' + '.pickle', 'rb') as f:
    #     saved_obj_new = pickle.load(f)

    has_step = my_redis.exists(scope + ':step')
    has_q_dicts = my_redis.exists(scope + ':q')
    if not has_step or not has_q_dicts:
        return None
    restore_dicts = dict()
    restore_dicts[scope + ':step'] = int(my_redis.get(scope + ':step').decode('utf-8'))
    q_dicts = my_redis.hgetall(scope + ':q')

    temp_dicts = dict()
    for pair in q_dicts.items():
        temp_dicts[pair[0].decode('utf-8')] = np.array([float(item) for item in pair[1].decode('utf-8').split('_')])
    restore_dicts[scope + ':q'] = temp_dicts

    return restore_dicts


r = redis.StrictRedis(host='127.0.0.1', port=6379, db=0, password='***')
scope = 'abc'

# q_learning_scope = 'q_learning'
#
# P = defaultdict(actions)
#
# print('start loading')
# with open(q_learning_scope + '.pickle', 'rb') as f:
#     saved_obj = pickle.load(f)
# print(saved_obj.keys())
# Q = saved_obj['q']
#
# print('start transfer')
# for key in Q:
#     P['_'.join(map(str, map(int, key)))] = '_'.join(['{:0.3f}'.format(item) for item in Q[key]])
# saved_obj['q'] = P
# print('len is {}'.format(len(saved_obj['q'])))
# print('start saving')
# store_redis(saved_obj['q'], saved_obj['step'], r, scope)

print('start loading temp')
saved_obj_new = restore_redis(r, scope)
print('new obj len is {}'.format(len(saved_obj_new[scope + ':q'])))

i = 0
for key in saved_obj_new[scope + ':q']:
    i = i + 1
    print(key)
    print(saved_obj_new[scope + ':q'][key])
    if i == 10:
        break
