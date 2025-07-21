
# CalicoAI COMP3000

## Abstract : 

Recent advances in Deep Reinforcement Learning have created board game AI, capable of beating even the highest skilled players. The most recent and notable example of this is AlphaZero’s mastery of Go, Chess and Shori with much of the mainstream media attention focusing on similar traditional, deterministic, abstract strategy games. However, many contemporary board games incorporate elements of luck and randomness to create more dynamic gameplay experiences. In this report, the effectiveness of a long-standing deep reinforcement learning method, Deep Q-learning, will be tested in a modern stochastic boardgame environment, the 2020 board game Calico. It will explore the mechanisms and parameters necessary to train a Deep Q-learning agent in such an environment. Discuss, and provide solutions to, stabilization issues faced.  As well as investigating multiple network architectures. By the end, the results suggest that a Deep Q-learning agent can generalize certain versions of the Calico environment, and produce a capable policy given the correct combination of state representation, stabilization methods, and reward function engineering.  The evaluation shows the final trained model can outperform an agent making arbitrary decisions by a significant margin, with the model even able to beat some novice human players. Remarks on future research suggest expanding the scope of the game environment, allowing the AI access to additional information, giving the AI a larger section of actions as well as possibly testing other deep reinforcement learning techniques in the Calico environment. 

## Features :

### Playable Deep Q-Learning Agent

![gameplay_gif](https://github.com/matthew-lowsley/Calico-AI/blob/main/readme_images/readme1.gif)

### Training Statistics

![graphs_img](https://github.com/matthew-lowsley/Calico-AI/blob/main/readme_images/readme2.PNG)

### Convolutional Architecture

![network_diagram](https://github.com/matthew-lowsley/Calico-AI/blob/main/readme_images/readme3.png)

## How to Install and Run :

1. Clone this repo

```
git clone --recursive https://github.com/matthew-lowsley/Calico-AI-matthew-lowsley.git
```

2. Create and open python virtual environment (optional, but recommended)

```
py -m pip install --user virtualenv
```
```
py -m venv venv
```
```
.\venv\Scripts\activate
```

3. Install packages

```
pip install -r requirements.txt
```

4. Run the program

```
py main.py
```

## Startup Options :

Select mode: Train or Play. Play is default.

```
--mode=[play,train]
```

Add a custom selection of player agents. Options: h=human-controlled, r=random-agent, q=dq-agent Format: q,h,r = 1 dq-agent, 1 human-controlled, 1 random. Maximum players is four.
Default is q,r,h.

```
--players=q,h,r
```

Select a pretrained model to use. In Train mode this option is None. In Play mode this option
is model-34.11-final-version.pth by default. Model must be in the models folder.

```
--model=model-34.11-final-version.pth
```

The amount of time in ms between each player's turn. Default is 2000 in play mode and 0 in
train mode. Custom amount must be greater than 0.

```
--speed=2000
```

When True no game graphics will be displayed.

```
--headless=True
```

When True statistics will plotted.

```
--graph=True
```






