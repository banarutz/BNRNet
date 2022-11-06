from imports import *
from VGG16 import VGGNet
from utils import to_numpy
import torch.nn 

def get_mse (predict, gt):
    mse_diff = np.mean((predict - gt) ** 2)
    return mse_diff	

def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss

img = cv2.imread("D:\\se_joaca_licenta\\Datasets\\CelebA\\Img\\images_cropped_png\\img_align_celeba_png\\000024.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
img_L = img[:,:,0]
img_A = img[:,:,1]
img_B = img[:,:,2]
img_L = img_L / 255.0
img_A = img_A / 255.0
img_B = img_B / 255.0


tran = transforms.ToTensor()
img_L = tran(img_L)
img_L.double
img_L = img_L[None, :, :, :]

model = VGGNet()

predict = model(img_L)
predict = to_numpy(predict)

predict_A = predict[0, 0, :, :]
# print(predict.shape)

mse = torch.nn.MSELoss ()
MSE = mse(tran(predict_A), tran(img_A))
loss = get_mse (predict_A, img_A)
MSE_loss = my_loss(tran(predict_A), tran(img_A))
loss_mse = mse_loss(tran(predict_A), tran(img_A))

print('Loss custom = ', MSE_loss, 'loss calculat cu formula = ', loss , 'MSE din torch.nn = ', MSE, 'mse_loss din torch.functional = ', loss_mse)
# print (img_L.shape, img_A.shape)
# exit()
# print(np.max(img_A), np.max(img_L))

print(get_mse(img_L, img_A), my_loss(tran(img_L), tran(img_A)))
