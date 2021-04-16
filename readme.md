## Ensemble-based Dynamics with Probabilistic NN for Noisy-Mnist

### Prerequisites

tensorflow-gpu 1.13 or 1.14,
tensorflow-probability 0.6.0,
openAI [baselines](https://github.com/openai/baselines),

### Usage 

The following command should train the Probabilistic Ensemble for "noise mnist" MDP.

```
python baseline_train.py
```

This command will train for 200 epochs. The weights of VDM saved in `model/`. Then use following command to perform the conditional generation process to produce the figures.
```
python baseline_generate.py
```
