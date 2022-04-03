# Introduction

This repository is for all my projects and experimentations with Machine Learning. <br>
Right now I have included the OpenAI Gym enviroments I've solved as well as my work on my Diploma Thesis titled: <br>
**Decomposing Complex Reinforcement Learning Tasks using Testworld**


## OpenAI Gym

For general information about what OpenAI Gym is you can refer here: [OpenAI Gym](https://www.gymlibrary.ml) <br>
For the purpose of running all notebooks (.ipynb) files you will first and foremost need to install Gym. 
More information on how to install Gym found here: [Gym Github repository](https://github.com/openai/gym)

### How to Run
After that, you're ready to run any of the 'OpenAI Gym' notebooks I have.
> **Note:** That is assuming you already have the Numpy, Matplotlib, Keras libraries

## Diploma Thesis

The purpose of this thesis is to deconstruct a complex user written command (in the future speech-to-text)
into its sub-parts which will be then fed into RL agents, making learning faster. <br>
The deconstruction takes place in a TextWorld enviroment which can simulate real-world enviroments (e.g. a house).
This is done using RL where a TextWorld agent is trained to reach a given goal.
The training in this abstract enviroment results in a series of steps (i.e. the sub-tasks) which will be fed in a more detailed enviroment
(MiniGrid) to guide a different RL agent for faster convergence.

### How to Run
You can see the current results(code cell outputs) of my thesis by simply opening any .ipynb file.
> **Note:** This is still a work in progress. Also I recommend checking the QLearning_Agent.ipynb.

Alternatively, if you want to run the notebooks yourself then you will need to install _Gym_(as mentioned above) and _TextWorld_.
The latter runs only on Linux and MacOS systems. If you use Windows (like me), there is a workaround using Docker.
More information on TextWorld found here: <br>
[TextWorld GitHub](https://github.com/microsoft/TextWorld) 

