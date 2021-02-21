import numpy as np
import tensorflow as tf

# sets, rows, columns
## 2D CNN expects a 4D (batch, height, width, channels)

inp=np.arange(75).reshape(1,5,5,3).astype(np.float32)

kernel = np.array([[ [1,1,1],[2,1,2],[1,1,1] ]]).astype(np.float32).reshape(1,3,3,1)

print('input shape ->',inp.shape)
print('kernel shape ->',kernel.shape)

result = tf.nn.conv2d(inp, kernel, strides=(1,1,1,1), padding='VALID')

[[      [[ 44.],
         [ 77.],
         [110.]],

        [[209.],
         [242.],
         [275.]],

        [[374.],
         [407.],
         [440.]],

        [[539.],
         [572.],
         [605.]],

        [[704.],
         [737.],
         [770.]]   ]]

####################################################################
import numpy as np
import tensorflow as tf

# sets, rows, columns
## 2D CNN expects a 4D (batch, height, width, channels)

inp=np.random.randint(0,2,7*7*3).reshape(1,7,7,3).astype(np.float32)

kernel = np.array([[1,1,1],[1,2,1],[1,1,1]]).astype(np.float32).reshape(1,3,3,1)

print('input shape ->',inp.shape)
print('kernel shape ->',kernel.shape)

result = tf.nn.conv2d(inp, kernel, strides=(1,1,1,1), padding='SAME')
result.shape
tf.reshape(result,[7,7])
