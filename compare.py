import numpy as np

c1 = input('Enter the name of the first file: ')
data1 = np.fromfile(c1, sep=',')

print('Good')
c2 = input('Now enter the name of the second file: ')
data2 = np.fromfile(c2, sep=',')

if np.all(data1 == data2):
    print('The two files are the same, congrats!')
else:
    print('The two files are different, there\'s a problem...')
