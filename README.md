# Reinforcement Learning Final Project - Pablo Salar Carrera

## Project Overview
This project explores the performance of Deep Q-Networks (DQN) agents trained with varying numbers of hidden layers and different hidden layer sizes. The aim was to assess how network complexity impacts training time and the agent’s learning capabilities.

## Approach
We experimented with different configurations of hidden layers, varying both the number of layers and the size of each layer. The DQN agent was trained on a chosen environment, and the results were evaluated based on learning performance and the time taken to train the agent.

## Key Findings

1. **Training Time**: As the number of hidden layers and the size of the layers increased, the training time also increased significantly. More complex networks took longer to process and update during training.
  
2. **Learning Performance**: Despite the longer training times, the agent’s learning performance did improve marginally with more hidden layers and larger layer sizes. The agent learned slightly better with deeper architectures, but the improvement did not justify the increased computational cost.

3. **Trade-off**: The trade-off between training speed and performance was apparent. The increased learning capability with more hidden layers was not enough to offset the slower training times. In some cases, simpler network architectures performed adequately and much faster than the deeper counterparts.

## Conclusion
- Increasing the complexity of the DQN agent (in terms of hidden layers and their sizes) leads to marginal improvements in learning performance.
- However, this improvement is not substantial enough to justify the increased training time.
- Future work could explore optimizing the number of layers and their sizes to find a more efficient trade-off between performance and computation.

## Folder Structure

The project is organized as follows:

- `/notebooks/`: Contains the Jupyter Notebook used for analysis.
  - `DS669FinalProject_PabloSalar.ipynb` - Jupyter notebook with code, visualizations, and explanations.

- `/src/`: Contains the Python script used in the project.
  - `DS669FinalProject_PabloSalar.py` - Standalone Python script for running the analysis.

- `/reports/`: Contains the final project documentation and results.
  - `DS669FinalProject_PabloSalar.html` - The complete project report in HTML format.
