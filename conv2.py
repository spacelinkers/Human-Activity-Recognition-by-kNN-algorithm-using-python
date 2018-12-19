# import function
from include import*


'''
def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)
'''

def conv2(x, y, mode='same'):
    return signal.convolve2d(x, np.rot90(y), mode='same')


