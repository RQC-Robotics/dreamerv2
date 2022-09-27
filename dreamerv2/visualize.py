import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings

try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import agent
import common


def main():
    configs = yaml.safe_load((pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
    parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
    config = common.Config(configs['defaults'])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = common.Flags(config).parse(remaining)

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / 'config.yaml')
    print(config, '\n')
    print('Logdir', logdir)

    import tensorflow as tf
    tf.config.experimental_run_functions_eagerly(not config.jit)
    message = 'No GPU found. To actually train on CPU remove this assert.'
    assert tf.config.experimental.list_physical_devices('GPU'), message
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        from tensorflow.keras import mixed_precision as prec
        prec.set_global_policy(prec.Policy('mixed_float16'))

    train_replay = common.Replay(logdir / 'train_episodes', **config.replay)
    eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
        capacity=config.replay.capacity // 10,
        minlen=config.dataset.length,
        maxlen=config.dataset.length))
    step = common.Counter(train_replay.stats['total_steps'])
    outputs = [
        common.TerminalOutput(),
        common.JSONLOutput(logdir),
        common.TensorBoardOutput(logdir),
    ]
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_train = common.Every(config.train_every)
    should_log = common.Every(config.log_every)
    should_video_train = common.Every(config.eval_every)
    should_video_eval = common.Every(config.eval_every)
    should_expl = common.Until(config.expl_until // config.action_repeat)

    def make_env(mode):
        suite, task = config.task.split('_', 1)
        if suite == 'dmc':
            env = common.DMC(
                task, config.action_repeat, config.render_size, config.dmc_camera,
                pn_number=config.pn_number
            )
            env = common.NormalizeAction(env)
        elif suite == 'atari':
            env = common.Atari(
                task, config.action_repeat, config.render_size,
                config.atari_grayscale)
            env = common.OneHotAction(env)
        elif suite == 'crafter':
            assert config.action_repeat == 1
            outdir = logdir / 'crafter' if mode == 'train' else None
            reward = bool(['noreward', 'reward'].index(task)) or mode == 'eval'
            env = common.Crafter(outdir, reward)
            env = common.OneHotAction(env)
        elif suite == 'rlbench':
            env = common.RLBenchEnv(
                task,
                action_repeat=config.action_repeat,
                size=config.render_size,
                pn_number=config.pn_number
            )
            env = common.NormalizeAction(env)
        else:
            raise NotImplementedError(suite)
        env = common.TimeLimit(env, config.time_limit)
        return env

    def per_episode(ep, mode):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
        logger.scalar(f'{mode}_return', score)
        logger.scalar(f'{mode}_length', length)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
        should = {'train': should_video_train, 'eval': should_video_eval}[mode]
        if should(step):
            for key in config.log_keys_video:
                logger.video(f'{mode}_policy_{key}', ep[key])
        replay = dict(train=train_replay, eval=eval_replay)[mode]
        logger.add(replay.stats, prefix=mode)
        logger.write()

    print('Create envs.')
    num_eval_envs = min(config.envs, config.eval_eps)
    if config.envs_parallel == 'none':
        if config.task.startswith('rlbench'):
            assert config.envs == 1
            train_envs = [make_env('train')]
            eval_envs = train_envs
        else:
            train_envs = [make_env('train') for _ in range(config.envs)]
            eval_envs = [make_env('eval') for _ in range(num_eval_envs)]
    else:
        make_async_env = lambda mode: common.Async(
            functools.partial(make_env, mode), config.envs_parallel)
        train_envs = [make_async_env('train') for _ in range(config.envs)]
        eval_envs = [make_async_env('eval') for _ in range(num_eval_envs)]
    act_space = train_envs[0].act_space
    obs_space = train_envs[0].obs_space
    train_driver = common.Driver(train_envs)
    train_driver.on_episode(lambda ep: per_episode(ep, mode='train'))
    train_driver.on_step(lambda tran, worker: step.increment())
    train_driver.on_step(train_replay.add_step)
    train_driver.on_reset(train_replay.add_step)
    eval_driver = common.Driver(eval_envs)
    eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
    eval_driver.on_episode(eval_replay.add_episode)

    prefill = max(0, config.prefill - train_replay.stats['total_steps'])
    if prefill:
        print(f'Prefill dataset ({prefill} steps).')
        random_agent = common.RandomAgent(act_space)
        train_driver(random_agent, steps=prefill, episodes=1)
    #     eval_driver(random_agent, episodes=1)
        train_driver.reset()
    #     eval_driver.reset()

    print('Create agent.')
    train_dataset = iter(train_replay.dataset(**config.dataset))
    agnt = agent.Agent(config, obs_space, act_space, step)
    train_agent = common.CarryOverState(agnt.train)
    train_agent(next(train_dataset))
    if (logdir / 'variables.pkl').exists():
        print('Preload agent.')
        agnt.load(logdir / 'variables.pkl')
    else:
        raise NotImplementedError
        print('Pretrain agent.')
        for _ in range(config.pretrain):
            train_agent(next(train_dataset))
    train_policy = lambda *args: agnt.policy(
        *args, mode='explore' if should_expl(step) else 'train')
    eval_policy = lambda *args: agnt.policy(*args, mode='eval')

    report_dataset = iter(train_replay.dataset(**config.dataset))
    eval_dataset = iter(eval_replay.dataset(**config.dataset))

    def train_step(tran, worker):
        if should_train(step):
            for _ in range(config.train_steps):
                mets = train_agent(next(train_dataset))
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            logger.add(agnt.report(next(report_dataset)), prefix='train')
            logger.write(fps=True)

    train_driver.on_step(train_step)
    eval_driver.reset()
    eval_driver(eval_policy, episodes=1)

    while step < config.steps:
        logger.write()
        print('Start evaluation.')
        logger.add(agnt.report(next(eval_dataset)), prefix='eval')
        import pdb; pdb.set_trace()
        data = next(eval_dataset)
        data = agnt.wm.preprocess(data)
        video = agnt.wm.video_pred(data, 'image')
        video = video['openl_image'].numpy()
        video = video[0]
        video = np.clip(255 * video, 0, 255).astype(np.uint8)

        from PIL import Image

        imgs = [Image.fromarray(img) for img in video]
        if video.shape[1] == 64:
            imgs = [img.resize((128, 128)) for img in imgs]
        imgs[0].save(f'{logdir}/imagination.gif', save_all=True,
                     append_images=imgs[1:], optimize=False, duration=len(imgs))
        break
    #     if config.task.startswith('rlbench'):
    #         eval_driver.reset()
    #     eval_driver(eval_policy, episodes=config.eval_eps)
    #     print('Start training.')
    #     if config.task.startswith('rlbench'):
    #         train_driver.reset()
    #     train_driver(train_policy, steps=config.eval_every)
    #     agnt.save(logdir / 'variables.pkl')
    # for env in train_envs + eval_envs:
    #     try:
    #         env.close()
    #     except Exception:
    #         pass

    def plot():
        import json
        import matplotlib.pyplot as plt

        step = -1
        steps, values = [], []
        for line in open(f'{logdir}/metrics.jsonl'):
            line = json.loads(line)
            if 'eval_return' in line.keys():
                if step < line['step']:
                    step = line['step']
                    steps.append(line['step'])
                    values.append(line['eval_return'])

        plt.figure(figsize=(10, 8))
        plt.title(config.task)
        plt.ticklabel_format(axis='x', style='scientific', scilimits=(0, 0))
        plt.ylim(ymin=0, ymax=1000)
        plt.plot(steps, values)
        plt.ylabel('eval_return')
        plt.xlabel('environment_steps')
        plt.savefig(f'{logdir}/eval_return.pdf', format='pdf', dpi=400)


    plot()


if __name__ == '__main__':
    main()