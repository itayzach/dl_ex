import matplotlib.pyplot as plt
import csv
import argparse
import pandas as pd

# Parse args
parser = argparse.ArgumentParser(description='Plot graphs')
parser.add_argument('--fname', help='csv filename path')
args = parser.parse_args()

# Get net name and optim
path = args.fname.split('/')
net, optim = path[-1].replace('.csv','').split('__')

# Read csv
df = pd.read_csv(args.fname, delimiter='\t')
max_epoch = df['test_acc'].idxmax()
test_acc_max = df['test_acc'].max()

# Plot
df.plot(title=net + ' with ' + optim + '\n' +
        'max accuracy at epoch ' + str(max_epoch) + ' is ' + str(test_acc_max) + '%')
plt.xlabel('Epoch')
plt.ylabel('Accuracy [%]')
plt.savefig('plots/' + net + '_' + optim + '.pdf')
