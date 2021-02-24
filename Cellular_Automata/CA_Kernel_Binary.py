import numpy as np
import itertools
regra=30

kernel=np.random.randint(2, size=(3,3))


def cellular_automaton():
    global kernel

    lista=[0,1]
    kernel=np.pad(kernel, (1, 1), 'constant', constant_values=(0))
    q12=np.array([p for p in itertools.product(lista, repeat=3)])[::-1]

    uau12 = np.zeros(q12.shape[0])
    temp = [int(i) for i in np.base_repr(int(regra),base=2)]
    uau12[-len(temp):]=temp
    ru12=np.array(range(0,len(uau12)))

    tod12=[]
    for i in range(0,len(uau12)):
        tod12.append([0,int(uau12[i]),0])

    final=[]
    for i in range(0,len(q12)):
        final.append(np.array([q12[i],np.array(tod12).astype(np.int8)[i]]))

    
    def ca(row):
        out=[]
        for xx in range(0,3):
            out.append(tod12[next((i for i, val in enumerate(q12) if np.all(val == kernel[row][xx:xx+3])), -1)][1])
        return out
    kernel=np.array([item for item in map(ca,range(1,kernel.shape[0]-1))])
    return kernel

cellular_automaton()
