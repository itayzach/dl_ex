import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Plot graphs')
parser.add_argument('--name', help='name of this run')
args = parser.parse_args()
if (not args.name):
    print('gotta have a name')
    exit()

#headers = ['Epochs','Test accuracy (%)','Train accuracy (%)']
df = pd.read_csv(args.name, delimiter='\t')
print (df)

x = df['epoch']
y1 = df['test_acc']
y2 = df['train_acc']

# plot
plt.plot(x,y1,y2)
plt.show()
