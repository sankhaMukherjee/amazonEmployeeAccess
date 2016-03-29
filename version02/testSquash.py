import numpy as np
import matplotlib.pyplot as plt
 
def atan(x, w=1, c=0):
    '''
        w = width parameter
        c = center
    '''
    x = x*2 -1
    v = np.arctan( 2*(x-c)/w )
    v = (v - v.min())/( v.max() - v.min() )
    return v
 
x = np.linspace(0, 1, 1000)
 
for w in np.logspace(-1, 1, 5):
    plt.figure(figsize=(4,3))
    plt.axes([0.14, 0.17, 0.94-0.14, 0.90-0.17])
    for c in [-0.75, -0.5, 0, 0.5, 0.75]:
        y = atan(x, w, c)
        plt.plot(x, y, color='salmon')
    plt.title('w = %.3f'%w)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.savefig('%.3f-%.3f.png'%(c, w))
 
plt.show()
 
