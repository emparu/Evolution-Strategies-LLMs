Link to Kaggle notebook: https://www.kaggle.com/code/emanuelruzak/evolutonstrategiesv2

Comment: The huggingface API key is not a working key.

<img width="1200" height="700" alt="Image" src="https://github.com/user-attachments/assets/cc4b373f-cb2d-4e00-b92b-8799dcd96994" />


In "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning" (Qiu et al., 2025), a method based on evolutionary strategies (ES) was introduced for reinforcement learning in LLMs.

The method, on each iteration, consists basically of generating K perturbed versions of an LLM by adding noise to its weights, calculating the reward for each perturbed version, and finally reweighting the perturbations by the z-score and adding them to the LLM.

This method is much simpler and more efficient than other RL methods like PPO and GRPO and can do general reinforcement learning unlike other "RL-like" methods like DPO or decision transformers, which are constrained to specific scenarios and don't allow the model to "learn from its own generations."

I am implementing this method on Gemma3-1B-it with the objective of maximizing accuracy on 3-digit multiplication.

For my version of ES, the variance of the noise I used depended on the variance of each layer, since the magnitude of the weights can vary wildly between layers.

In my experiment I used 13 generations to evaluate the reward for each perturbed model, 13 perturbed models per iteration, and 65 evolution steps. The experiment took 10 hours on a Kaggle 2x Tesla T4 VM. The accuracy went from ~15% to ~60%.

The results were surprising, since it's hard to get RL to work in such a short time. As a comparison, GRPO for simple tasks on a 0.5B model could take as long as 24 hours on 8 A100 GPUs. https://huggingface.co/docs/trl/main/en/grpo_trainer

Another advantage is that ES requires much less hyperparameters than GRPO.

Another thing to mention is that the performance on the multiplication task I tested ES on was very dependent on prompting, and it is possible that the task was very easy for "RL-ing," and thus the performance reached in my experiment doesn't reflect the performance of ES in general.
