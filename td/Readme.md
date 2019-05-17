The way to run td algorithm is:
```
cd td
nohup python -u q_learning.py --size 2 --outputInterval 100 > nohup.out 2> nohup.err &
```

or run with docker:
```
nohup sudo docker run --rm --network=host -v ~/Developer/Python/vue-2048-backend:/tmp -v ~/tensorboard:/tensorboard -w /tmp/td my-tensorflow python -u q_learning.py --size 3 --outputInterval 20000 --summaries_dir /tensorboard > q_learning.out 2>&1 &

```