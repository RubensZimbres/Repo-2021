import numpy as np
import itertools
regra=2159062512564987644819455219116893945895958528152021228705752563807958532187120148734120
base1=5
states=np.arange(0,base1)
dimensions=1
kernel=np.random.randint(len(states), size=(dimensions,100))

def cellular_automaton():
    global kernel

    lista=states
    kernel=np.pad(kernel, (1, 1), 'constant', constant_values=(0))
    q12=np.array([p for p in itertools.product(lista, repeat=3)])[::-1]

    uau12 = np.zeros(q12.shape[0])
    temp = [int(i) for i in np.base_repr(int(regra),base=base1)]
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
        for xx in range(0,100):
            out.append(tod12[next((i for i, val in enumerate(q12) if np.all(val == kernel[row][xx:xx+3])), -1)][1])
        return out
    kernel=np.array([item for item in map(ca,range(1,kernel.shape[0]-1))])
    return kernel

output=[]
for i in range(0,100):
    output.append(cellular_automaton())

output=np.array(output).reshape(100,100)
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axes_grid1

fig = plt.figure()
grid = axes_grid1.AxesGrid(
    fig, 111, nrows_ncols=(1, 2), axes_pad = 0.5, cbar_location = "right",
    cbar_mode="each", cbar_size="8%", cbar_pad="5%",)

im0 = grid[0].imshow(np.array(output), cmap='gray', interpolation='nearest')
grid.cbar_axes[0].colorbar(im0)

im1 = grid[1].imshow(np.array(output), cmap='jet', interpolation='nearest')
grid.cbar_axes[1].colorbar(im1)

plt.savefig('/home/theone/Pictures/CA1D_5_.png', bbox_inches='tight', pad_inches=0.0, dpi=200,)
