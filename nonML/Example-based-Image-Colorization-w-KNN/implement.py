import cv2
import skimage
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from skimage.segmentation import slic
from skimage.util import img_as_float
from scipy import ndimage as nd
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# This function extracts features based on gaussian (sigma = 3 and sigma = 7) and
# variance (size = 3)
def extract_all(img):

    img2 = img.reshape(-1)
    
    # First feature is grayvalue of pixel
    df = pd.DataFrame()
    df['GrayValue(I)'] = img2 

    # Second feature is GAUSSIAN filter with sigma=3
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

    # Third feature is GAUSSIAN fiter with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3    

    # Third feature is generic filter that variance of pixel with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img1
    
    return df

# This function extracts average pixel value of some neighbors
# frame size : (distance * 2) + 1 x (distance * 2) + 1
#default value of distance is 8 if the function is called without second parameter
def extract_neighbors_features(img,distance = 8):

    height,width = img.shape
    X = []

    for x in range(height):
        for y in range(width):
            neighbors = []
            for k in range(x-distance, x+distance +1 ):
                for p in range(y-distance, y+distance +1 ):
                    if x == k and p == y:
                        continue
                    elif ((k>0 and p>0 ) and (k<height and p<width)):
                        neighbors.append(img[k][p])
                    else:
                        neighbors.append(0)

            X.append(sum(neighbors) / len(neighbors))

    return X

# This function extracts superpixels
# Every cell has a value in superpixel frame so 
# It is extracting superpixel value of every pixel
def superpixel(image,status):    
    if status:
        segments = slic(img_as_float(image), n_segments = 100, sigma = 5)
    else:
        segments = slic(img_as_float(image), n_segments = 100, sigma = 5,compactness=.1) 

    return segments

# Function to calculate Mean Absolute Error
def calculate_mae(y_true,y_predict):
    
    # Calculate mean absolute error for every color according to MAE formula
    error_b = float(sum([abs(float(item_true) - float(item_predict)) for item_true, item_predict in zip(y_true[:,0], y_predict[:,0])]) / len(y_true))
    error_g = float(sum([abs(float(item_true) - float(item_predict)) for item_true, item_predict in zip(y_true[:,1], y_predict[:,1])]) / len(y_true))
    error_r = float(sum([abs(float(item_true) - float(item_predict)) for item_true, item_predict in zip(y_true[:,2], y_predict[:,2])]) / len(y_true))
    
    # Return aveage of colours error
    return (((error_b + error_g + error_r) / 3))

# Function to save predicted images in Outputs folder in Dataset folder
def save_picture(test_data,rgb_data_name,y_predict):
    
    # If Outputs folder is not exist in directory of Dataset create it
    if not os.path.exists('Example-based-Image-Colorization-w-KNN/Outputs'):
        os.makedirs('Example-based-Image-Colorization-w-KNN/Outputs')
    
    # Create an array for colorful image 
    height,width = test_data.shape
    data = np.zeros((height, width, 3), dtype=np.uint8)

    # Fill the data with predicted RGB values
    tmp = 0
    for i in range(height):
        for k in range(width):
            data[i,k] = [y_predict[tmp][0], y_predict[tmp][1], y_predict[tmp][2]]
            tmp +=1
            
    # Save predicted image
    cv2.imwrite('Outputs/' + rgb_data_name + '.jpg', data)
    return data 

distance = 1
minImageIndex  = 1
maxImageIndex  = 50

# Read Images from Dataset folder

train_data_PATH = os.path.join('/home/intern1/Desktop/BNRNet/nonML/Example-based-Image-Colorization-w-KNN/Example-based-Image-Colorization-w-KNN/source_images/')
train_data_names = os.listdir(train_data_PATH)

test_data_PATH = os.path.join('/home/intern1/Desktop/BNRNet/nonML/Example-based-Image-Colorization-w-KNN/Example-based-Image-Colorization-w-KNN/target_images/')
accuracy_data_PATH = os.path.join('/home/intern1/Desktop/BNRNet/nonML/Example-based-Image-Colorization-w-KNN/Example-based-Image-Colorization-w-KNN/gt_images/')

# rgb_data using for source colorful images
rgb_data = [cv2.imread(train_data_PATH + file, 1) for file in (train_data_names)]

# train_data_names is using for image names
# train_data_names = [file.split(',')[0].split('\\')[1] for file in glob.glob("images/*source.png")][minImageIndex-1:maxImageIndex]

# train_data is grayscale of colorful source images
train_data = [cv2.imread(train_data_PATH + file, 0) for file in (train_data_names)]


# test_data is using for grayscale type of target images
test_data = [cv2.imread(test_data_PATH + file, 0) for file in (train_data_names)]

# accuracy_data is using for colorful groundtruth images

accuracy_data = [cv2.imread(accuracy_data_PATH + file, 1) for file in (train_data_names)]

# Create an array for Mean Absolute Error
MAE = []

# Create a file with name ProjectTestImagesResults.txt for saving MAE values in it
f = open("ProjectTestImagesResults.txt", "w")


print(len(train_data))

for i in range(len(train_data)):   
    print(i)
    # preparing y (b, g, r)
    y = rgb_data[i].reshape((-1,3))
    
    # preparing y_true (b, g, r)
    y_true = accuracy_data[i].reshape((-1,3))       
    
    # preparing X variable
    X1 = extract_all(train_data[i]).values
    X2 = superpixel(train_data[i],False).reshape(-1,1)
    X3 = extract_neighbors_features(train_data[i],distance)
    X = np.c_[X1, X2, X3]    
    # Now we have input for training the model 
    # We have total 6 feature in X which are
    #Colums: GrayValue, Gaussians3, Gaussians7, GenericFilter(variance)s3, superpixel, averageOfFrameGrayValues    
    
    
    # preparing X_test variable
    X1_test = extract_all(test_data[i]).values
    X2_test = superpixel(test_data[i],False).reshape(-1,1)
    X3_test = extract_neighbors_features(test_data[i],distance)
    X_test = np.c_[X1_test, X2_test, X3_test]
    # Now we have input X_test values too for predict RGB values from it
    
    # training model
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X,y)
    
    # testing model
    y_predict = knn_clf.predict(X_test)
    
    # Calculate accuracy score
    MAE.append(calculate_mae(y_true,y_predict))       

    # Save Picture to Dataset/Outputs folder
    predicted_picture = save_picture(test_data[i],train_data_names[i],y_predict)  
    
    # Plot Original and predicted images
    # fig, ax = plt.subplots(1, 2 , figsize=(16,8))
    # ax[0].imshow(cv2.cvtColor(accuracy_data[i], cv2.COLOR_BGR2RGB))
    # ax[0].set_title("Original Image of " + train_data_names[i])

    # ax[1].imshow(cv2.cvtColor(predicted_picture, cv2.COLOR_BGR2RGB))
    # ax[1].set_title("Predicted Image of " + train_data_names[i])
    # plt.show()
    
    print("Mean Absolute Error of",train_data_names[i],"is",MAE[i]) 
    f.write(str(MAE[i]))
    f.write('\n')
    print( '#####################################################')