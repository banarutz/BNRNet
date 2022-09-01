from imports import *
from utils import *

def main():

    img = [[1,2,3,4,5],
            [1,2,3,4,5]]

    img = np.array (img)
    add_noise(img, 0)

if __name__ == '__main__':
    main()
