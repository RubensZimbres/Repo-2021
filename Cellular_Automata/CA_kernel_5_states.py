import numpy as np
import itertools
regra=2159062512564987644819455219116893945895958528152021228705752563807958532187120148734120


lista=[0,1,2,3,4]

q12=np.array([p for p in itertools.product(lista, repeat=3)])[::-1]

uau12 = np.base_repr(int(regra),base=5)

ru12=np.array(range(0,len(uau12)))

tod12=[]
for i in range(0,len(uau12)):
    tod12.append([0,int(uau12[i]),0])

final=[]
for i in range(0,len(q12)):
    final.append(np.array([q12[i],np.array(tod12).astype(np.int8)[i]]))

import matplotlib.pyplot as plt

_, axs = plt.subplots(5, 25, figsize=(12, 5))
axs = axs.flatten()
for img, ax in zip(final, axs):
    ax.imshow(img, cmap='gray')
    ax.grid(False)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_yticklabels('')
    ax.set_xticklabels('')
plt.show()

kernel=np.random.randint(2, size=(3,3))

kernel=np.pad(kernel, (1, 1), 'constant', constant_values=(0))

def ca(row):
    out=[]
    out.append(0)
    for xx in range(0,3):
        out.append(tod12[next((i for i, val in enumerate(q12) if np.all(val == kernel[row][xx:xx+3])), -1)][1])
    out.append(0)
    return out

for i in range(0,kernel.shape[0]):
    ca(i)
