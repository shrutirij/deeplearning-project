import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 12})

labels = ['Full', 'Fixed', 'Small']
filenames = list(sys.argv[1].split(','))
max_eps = int(sys.argv[2])

for i,filen in enumerate(filenames):
    epochs = []
    val_acc = []

    with open(filen, 'r') as f:
        for line in f:
            if 'Epoch' in line:
                epochs.append(int(line.strip().split()[-1]))
                x=0
            if 'Validation loss' in line and x == 0:
                val_acc.append(float(line.strip().split()[-1]))
                x=1

    plt.plot(epochs[:max_eps+1], val_acc[:max_eps+1], label=labels[i])
plt.xlabel('Epochs')
plt.ylabel('Validation Total Cross Entropy Loss')
plt.legend()
plt.savefig('val_loss.pdf', bbox_inches='tight')
