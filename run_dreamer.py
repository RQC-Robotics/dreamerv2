import os
import shlex
import argparse
import subprocess
import itertools

os.environ['MUJOCO_GL'] = 'egl'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

config_profile = dict(
    point_cloud='dmc_point_cloud',
    image='dmc_vision'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str)
    parser.add_argument('--observe', type=str, nargs='+')
    parser.add_argument('--tasks', type=str, nargs='+')
    return parser.parse_args()


def run_job(logdir, observation, task):
    os.makedirs(logdir, exist_ok=False)
    stderr = open(f'{logdir}/stderr.txt', 'w')
    stdout = open(f'{logdir}/stdout.txt', 'w')
    command = (
        f'python dreamerv2/train.py',
        f'--logdir {logdir}',
        f'--configs {config_profile[observation]}',
        f'--task {task}'
    )
    command = ' '.join(command)
    return subprocess.Popen(shlex.split(command), stderr=stderr, stdout=stdout)


if __name__ == "__main__":
    args = parse_args()
    jobs = itertools.product(args.observe, args.tasks)
    log_fn = lambda o, t: f'{args.logdir}/{t}/{o}'
    procs = []
    for obs, task in jobs:
        logdir = log_fn(obs, task)
        procs.append(run_job(logdir, obs, task))

