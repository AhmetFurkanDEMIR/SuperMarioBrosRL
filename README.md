 ![](https://img.shields.io/badge/microsoft%20azure-0089D6?style=for-the-badge&logo=microsoft-azure&logoColor=white) ![](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white) ![](https://img.shields.io/badge/NVIDIA-Tesla%20K80-76B900?style=for-the-badge&logo=nvidia&logoColor=white) ![](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white) ![](https://img.shields.io/badge/TensorFlow%20-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white)

# Ray RLlib - Super Mario Bros 

![gf](https://user-images.githubusercontent.com/54184905/118782592-efa36480-b896-11eb-91b6-057d4521fd7a.gif)

Using the  [DDDQN (Dueling Double Deep Q Learning)](https://docs.ray.io/en/master/rllib-algorithms.html#dqn) algorithm with Ray['RLlib'] on the gym-super-mario-bros environment to make the mario character finish the game by itself.

**What is Ray? :** [Ray](https://ray.io/) provides a simple, universal API for building distributed applications. [Ray](https://ray.io/) accomplishes this mission by:

- Providing simple primitives for building and running distributed applications.

- Enabling end users to parallelize single machine code, with little to zero code changes.

- Including a large ecosystem of applications, libraries, and tools on top of the core [Ray](https://ray.io/) to enable complex applications.


**What is RLlib? :** Scalable Reinforcement Learning, [RLlib](https://docs.ray.io/en/master/rllib.html) is an open-source library for reinforcement learning that offers both high scalability and a unified API for a variety of applications. [RLlib](https://docs.ray.io/en/master/rllib.html) natively supports TensorFlow, TensorFlow Eager, and PyTorch, but most of its internals are framework agnostic.


**What is gym super mario bros? :** An OpenAI Gym environment for [Super Mario Bros. & Super Mario Bros. 2](https://pypi.org/project/gym-super-mario-bros/) (Lost Levels) on The Nintendo Entertainment System (NES) using the nes-py emulator. 


## Requirements

**Python3 libraries**

```console
sudo apt-get install python3-pip
sudo apt-get install python3-dev
pip3 install tensorflow-gpu
pip3 install ray
pip3 install ray['rllib']
pip3 install gym
pip3 install gym-super-mario-bros
```


**NVIDIA CUDA :**  [Setup document](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), Warning: If Tensorflow cannot establish a GPU connection via CUDA, the code will run on the CPU.


## Train

The following command starts training with the [DDDQN (Dueling Double Deep Q Learning)](https://docs.ray.io/en/master/rllib-algorithms.html#dqn) algorithm in the "SuperMarioBros-v0" environment, which comes by default in the [configPy.py](/configPy.py) file. For any changes, go to the [configPy.py](/configPy.py) file.

```console
python3 train.py
```

| Agent | Iteration | Steps | Max Reward | Min Reward | Mean Reward |
| ------|-----------|-------|------------|------------|------------ |
| DQN   |       616 | 617000|     17504  |     -5708 |     9048.6  |
| DQN   |       617 | 618000|     19847  |     -5708 |     9628.2  |
| DQN   |       ... |  ...  |      x     |      y     |       z     |

|        gym env        |         State       |  World | gif |
| ----------------------|---------------------|--------|-----|
|   SuperMarioBros-v0   |  Training process   |  1-1   | ![](https://user-images.githubusercontent.com/54184905/118872836-b0eec800-b8f1-11eb-9ee9-a887c74a0e1c.gif)|


## Test

You can test in "SuperMarioBros-v0" environment using weights trained with the command below.

```console
python3 test.py checkpoint-xxxx --env super_mario_bros --steps y
```

Sample command

```console
python3 test.py /home/demir/Desktop/rl/root/checkpoint_001501/checkpoint-1501 --env super_mario_bros --steps 2000
```

|        gym env        |         State       | World | Video |
| ----------------------|---------------------|-------|-----|
|   SuperMarioBros-v0   |        Test         |  1-1  | ![ezgif com-resize](https://user-images.githubusercontent.com/54184905/119223296-38f6ec80-bb01-11eb-8b8b-de01d7dd9f60.gif)|
|   SuperMarioBros-v0   |        Test         |  1-2  | None |
|   SuperMarioBros-v0   |        Test         |  1-3  | None |
|   SuperMarioBros-v0   |        Test         |  1-4  | None |


### Resources

[Link 1](https://docs.ray.io/en/master/rllib.html) | [Link 2](https://github.com/ray-project/ray/blob/master/rllib/rollout.py) | [Link 3](https://github.com/Kautenja/gym-super-mario-bros) | [Link 4](https://github.com/uvipen/Super-mario-bros-A3C-pytorch/blob/master/src/env.py) | [Link 5](https://towardsdatascience.com/marios-gym-routine-6f095889b207?source=social.tw) | [Link 6](https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial)

