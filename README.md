# Model Merging Technique and Implementation

Model merging is a cost-effective way to combine multiple models trained on different downstream tasks, giving the combined abilities of each model without any additional training.

We will implement the merging operation using the mergekit library. [Mergekit](https://github.com/arcee-ai/mergekit)

Model merging refers to combining multiple distinct Large Language Models (LLMs) into a single unified LLM without requiring additional training or fine-tuning. The primary goal behind this approach is that different LLMs, each optimized or fine tuned at specific tasks, can be merged to enhance performance across the combined expertise of both models or multiple models.  

## Merging Techniques
![image](https://github.com/GouthamVicky/Model-Merging/assets/65328702/da601d9a-b88d-4c3e-87f6-49d9dd683dad)

### Linear Merging ( Model Soup)
The classic merge method - a simple weighted average.
The linear method uses a weighted average to combine two or more models, with the weight parameter allowing users to precisely control the contribution of each model's characteristics to the final merged model.

Parameters:

weight - relative (or absolute if normalize=False) weighting of a given tensor

normalize - if true, the weights of all models contributing to a tensor will be normalized.

Two types of model soup

- Uniform soup : Average all models
- Greedy Soup : Average models one by one, keeping only the ones that gradually improve test accuracy

### SLERP ( Spherical Linear Interpolation)

![image](https://github.com/GouthamVicky/Model-Merging/assets/65328702/96131642-87e3-4538-8238-efe1b2d0b8de)

SLERP addresses the limitations of traditional weight averaging in model merging. It offers a more nuanced approach, blending models in a way that preserves the unique characteristics and curvature of each parent model in high-dimensional spaces.

It can be applied to only 2 models at a time.

Three steps on SLERP

- Normalization
- Angle Calculation
- Vector Weighing and Summation

### Task Arithmetic

This method introduces a new way for modifying the behavior of the model using “task vectors.” These vectors represent directions in the weight space of a pre-trained model, pointing towards improved performance on a specific task.

Vectors can be manipulated through arithmetic operations like negation and addition, allowing for targeted behavior changes in the model


![image](https://github.com/GouthamVicky/Model-Merging/assets/65328702/71629b8f-d711-41eb-8618-1e8b1b577694)

#### Advantages of using Task Arithmetic merging
- Negation to Decrease Performance: Negating a task vector diminishes the model’s performance on the target task while maintaining its behavior on control tasks.
- Addition for Multi-Task Improvement: Adding task vectors can enhance the model’s performance across multiple tasks simultaneously
- Analogical Task Improvement: Combining task vectors from related tasks can improve performance on a fourth task, even without using data from this task.

### TRIM, ELECT SIGN & MERGE (TIES-Merging)

TIES-merging is currently the most popular model merging method in the LLM community due to its ability to merge more than two models simultaneously.

The TIES-Merging method utilizes the task arithmetic framework to efficiently combine multiple task-specific models into a single multitask model, addressing the challenges of parameter interference and redundancy

The TIES-Merging method minimizes the loss of valuable information due to redundant parameter values and sign disagreements across models, which specifies task vectors and applies a sign consensus algorithm.


![image](https://github.com/GouthamVicky/Model-Merging/assets/65328702/bad64d22-7649-458b-aa6e-2e971a7e7997)

- Resetting Parameters: It resets parameters that have only changed marginally during fine-tuning. This step helps in reducing redundancy.
- Resolving Sign Conflicts: It resolves conflicts arising from differing signs of parameter values across models.
- Selective Merging: It only merges parameters that align with the final agreed-upon sign.

### DARE TIES/Task Arithmetic ( DROP AND RESCALE)

DARE primarily focuses on learning the parameters of similar models to gain new capabilities.

It uses a similar approach to TIES with two main differences:

Pruning of Delta Parameters: identifies and eliminates most delta parameters (the differences between fine-tuned and pre-trained parameters) by setting them to zero.

Rescaling Weights: includes a rescaling step where the weights of the models are adjusted to keep the output expectations approximately unchanged. This involves adding the rescaled weights of the models to the weights of the base model with a scale factor.


![image](https://github.com/GouthamVicky/Model-Merging/assets/65328702/1ebf6275-cb8c-4d53-aec9-f1d6c566954d)

- Pruning: resets fine-tuned weights to their original pre-trained values, reducing unnecessary parameter changes.
- Merging: averages parameters from multiple models to create a single, unified model.
- Rescaling: adjusts the merged model’s weights to preserve its expected performance.

### FrankenMerges (Passthrough)
This method concatenates layers from different models, enabling the creation of models with a unique number of parameters, such as combining two 7B models to form a 9B model

A passthrough is a no-op that simply passes input tensors through unmodified. It is meant to be used for layer-stacking type merges where we have only one input model.

Currently, this is the only method in Mergekit that works for different model architectures. This is because it doesn’t fuse different layers into a single one as other methods do, and instead just stacks different layers sequentially.


## Model Merging Experiment 

Combining two different models: **OpenMath-Mistral-7B-v0.1-hf** and **Mistral-7B-Merge-14-v0.1** and merge them with the SLERP method


This is an experiment to test merging 14 models using DARE TIES 🦙

The merged model is then merged again with [OpenMath](https://huggingface.co/nvidia/OpenMath-Mistral-7B-v0.1-hf) using Gradient SLERP.
The result is a base model that performs quite well but requires some further instruction fine-tuning.

The 14 models are as follows:
1. [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
2. [ehartford/dolphin-2.2.1-mistral-7b](https://huggingface.co/ehartford/dolphin-2.2.1-mistral-7b)
3. [SciPhi/SciPhi-Mistral-7B-32k](https://huggingface.co/SciPhi/SciPhi-Mistral-7B-32k)
4. [ehartford/samantha-1.2-mistral-7b](https://huggingface.co/ehartford/samantha-1.2-mistral-7b)
5. [Arc53/docsgpt-7b-mistral](https://huggingface.co/Arc53/docsgpt-7b-mistral)
6. [berkeley-nest/Starling-LM-7B-alpha](https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha)
7. [Q-bert/MetaMath-Cybertron-Starling](https://huggingface.co/Q-bert/MetaMath-Cybertron-Starling)
8. [Open-Orca/Mistral-7B-OpenOrca](https://huggingface.co/Open-Orca/Mistral-7B-OpenOrca)
9. [v1olet/v1olet_marcoroni-go-bruins-merge-7B](https://huggingface.co/v1olet/v1olet_marcoroni-go-bruins-merge-7B)
10. [beowolx/MistralHermes-CodePro-7B-v1](https://huggingface.co/beowolx/MistralHermes-CodePro-7B-v1)
11. [TIGER-Lab/MAmmoTH-7B-Mistral](https://huggingface.co/TIGER-Lab/MAmmoTH-7B-Mistral)
12. [teknium/OpenHermes-2.5-Mistral-7B](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B)
13. [Weyaxi/OpenHermes-2.5-neural-chat-v3-3-Slerp](https://huggingface.co/Weyaxi/OpenHermes-2.5-neural-chat-v3-3-Slerp)
14. [mlabonne/NeuralHermes-2.5-Mistral-7B](https://huggingface.co/mlabonne/NeuralHermes-2.5-Mistral-7B)

- base model: [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)

### Model links

- [OpenMath](https://huggingface.co/nvidia/OpenMath-Mistral-7B-v0.1-hf)
- [EmbeddedLLM](https://huggingface.co/EmbeddedLLM/Mistral-7B-Merge-14-v0.1)


The yaml config file for this model is here:

```yaml
slices:
  - sources:
      - model: EmbeddedLLM/Mistral-7B-Merge-14-v0
        layer_range: [0, 32]
      - model: nvidia/OpenMath-Mistral-7B-v0.1-hf
        layer_range: [0, 32]
merge_method: slerp
base_model: EmbeddedLLM/Mistral-7B-Merge-14-v0
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5
dtype: bfloat16

```
## Interactive Examples

You can run the interactive examples on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_2wF-X67Pc0ezfReNbU_AsIt5mqudzv_#scrollTo=RcvyWfFEaq3o)

## Merged Model Card Link 

[Goutham-Vignesh/OpenMath-Mistral-7B-Merge-14-v0.1](https://huggingface.co/Goutham-Vignesh/OpenMath-Mistral-7B-Merge-14-v0.1)

## References

- [Model Merging](https://blog.premai.io/model-merging/)
- [Merge Large Language Models with mergekit](https://towardsdatascience.com/merge-large-language-models-with-mergekit-2118fb392b54)
- [Merge Large Language Models](https://slgero.medium.com/merge-large-language-models-29897aeb1d1a)
- 
