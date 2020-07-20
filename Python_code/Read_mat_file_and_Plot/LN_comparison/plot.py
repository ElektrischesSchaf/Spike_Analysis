import matplotlib.pyplot as plt
import json
import os


# Plot the training results 
with open( 'with_LN.json', 'r') as f:
    with_LN = json.loads(f.read())

with open( 'no_LN.json', 'r') as f:
    no_LN = json.loads(f.read())

# train_loss = [l['loss'] for l in history['train']]
test_loss_1 = [l['loss'] for l in with_LN['test']]
test_loss_2 = [l['loss'] for l in no_LN['test']]
# train_R_square = [l['R^2'] for l in history['train']]
# valid_R_square = [l['R^2'] for l in history['test']]

my_fontsize=30

plt.figure(figsize=(16,9))

plt.plot(test_loss_1, 'b', linewidth=5, label = 'L.N.')
plt.plot(test_loss_2, 'g', linewidth=5, label = 'Original')

plt.xlabel('Epoch' , fontsize=my_fontsize*0.8 )
plt.xticks(fontsize=my_fontsize*0.8)
plt.ylabel('Loss', fontsize=my_fontsize*0.8 )
plt.yticks(fontsize=my_fontsize*0.8)
plt.legend( loc='upper right', fontsize=my_fontsize*0.8 )

plt.xlim([0, 60])

plt.tight_layout()
plt.savefig( 'Comparison.png' )

plt.cla()
plt.clf()
plt.close()