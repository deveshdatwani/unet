import pickle
import numpy as np
from matplotlib import pyplot as plt

with open('model-loss.pkl', 'rb') as file:
    modelLoss = pickle.load(file)

with open('attention-model-loss.pkl', 'rb') as file:
    attentionModelLoss = pickle.load(file)
    
plt.figure(figsize=(5,5))
plt.plot(list(range(len(modelLoss))), modelLoss, color='#F44336', label='vanilla unet')
plt.plot(list(range(len(attentionModelLoss))), attentionModelLoss, color='#00ACC1', label='unet with attention gates')
plt.legend()
plt.xlabel('Batch Id')
plt.ylabel('Dice Loss')
plt.yticks(np.arange(0, 1, 0.1))
plt.show()

