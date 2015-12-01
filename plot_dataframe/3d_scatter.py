import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

data = pd.DataFrame(np.random.randn(50,3))

data.columns = ['x','y','z']


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')


colors = pd.Series(np.random.randint(3, size=50))


data['c'] = colors

cmap = {1:'r',0:'b',2:'g'}

cc = data.apply(lambda x:cmap[x[3]], axis=1)

data['cc'] = cc

for c in list('rbg'):
    ax.scatter(data[data.cc == c]['x'],data[data.cc == c]['y'],data[data.cc == c]['z'], c = data[data.cc == c]['cc'])

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
