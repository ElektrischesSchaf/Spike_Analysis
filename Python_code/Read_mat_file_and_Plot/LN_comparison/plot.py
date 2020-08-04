import matplotlib.pyplot as plt
import json
import os

my_fontsize=30


'''
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



plt.figure(figsize=(16,9))

plt.plot(test_loss_1, 'b', linewidth=5, label = 'LN-GRU')
plt.plot(test_loss_2, 'g', linewidth=5, label = 'GRU')

plt.xlabel('Epoch' , fontsize=my_fontsize*0.8 )
plt.xticks(fontsize=my_fontsize*0.8)
plt.ylabel('Loss', fontsize=my_fontsize*0.8 )
plt.yticks(fontsize=my_fontsize*0.8)
plt.legend( loc='upper right', fontsize=my_fontsize*0.8 )

# plt.xlim([0, 60])

plt.tight_layout()
plt.savefig( 'Comparison_1.png' )

plt.cla()
plt.clf()
plt.close()
'''

plt.figure(figsize=(16,9))
FILE_PATH = './no/'
ALL_List_FILE = os.listdir(FILE_PATH)
ALL_List_FILE.sort()
List_FILE=ALL_List_FILE[:]
session_file_list=List_FILE

for session_k in range(len(session_file_list)):
    session_name=str(session_file_list[session_k])
    file_name_1 = './no/'+ session_name +'/csv_files/'  +'history.json'
    with open( file_name_1, 'r') as f:
        no_LN = json.loads(f.read())
    test_loss_2 = [l['loss'] for l in no_LN['test']]
    plt.plot(test_loss_2, 'g', linewidth=3, alpha=0.5,label = 'GRU')

FILE_PATH = './yes/'
ALL_List_FILE = os.listdir(FILE_PATH)
ALL_List_FILE.sort()
List_FILE=ALL_List_FILE[:]
session_file_list=List_FILE

for session_k in range(len(session_file_list)):
    session_name=str(session_file_list[session_k])
    file_name_1 = './yes/'+ session_name +'/csv_files/'  +'history.json'
    with open( file_name_1, 'r') as f:
        with_LN = json.loads(f.read())
    test_loss_1 = [l['loss'] for l in with_LN['test']]
    plt.plot(test_loss_1, 'b', linewidth=3, alpha=0.5, label = 'LN-GRU')

plt.xlabel('Epoch' , fontsize=my_fontsize*0.8 )
plt.xticks(fontsize=my_fontsize*0.8)
plt.ylabel('Loss', fontsize=my_fontsize*0.8 )
plt.yticks(fontsize=my_fontsize*0.8)
plt.xlim([0, 70])

# plt.legend( loc='upper right', fontsize=my_fontsize*0.8 )
# https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=my_fontsize*0.8 )

plt.tight_layout()
plt.savefig( 'Comparison_2.png' )