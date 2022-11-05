from VGG16 import VGGNet
from imports import *
from utils import *
from config import LOAD_MODEL_PATH, INFER_PATH

np.set_printoptions(threshold=sys.maxsize)

model = VGGNet()
# ckpt = torch.load(LOAD_MODEL_PATH + '11.pth' )
# model.load_state_dict(ckpt['state_dict'], strict=True)

infer_image = cv2.imread(INFER_PATH + '005115.png')
h, w, _ = infer_image.shape
infer_image_LAB = cv2.cvtColor(infer_image, cv2.COLOR_BGR2LAB)

# infer_image = cv2.resize(infer_image, (178, 218))
# show_img(infer_image[:,:, 1])

####### DE CE VREA RESIZE??? ##############

L_infer_image = infer_image[:, :, 0]
A_infer_image = infer_image[:, :, 1]
B_infer_image = infer_image[:, :, 2]
L_infer_image_numpy = L_infer_image

to_tensor = transforms.ToTensor()
L_infer_image = to_tensor(L_infer_image)
L_infer_image = L_infer_image[None, :, :, :]

predict = model(L_infer_image)

# predict = predict [:, :, :, :]
# predict = to_numpy(predict)
# predict = np.moveaxis(predict, 0, -1)
# show_img (predict)
# print(L_infer_image_numpy.shape, predict.shape)
# L_infer_image =

predict_A= predict[0, 0, :,:]
predict_B = predict[0, 1, :, :]

"""
A_infer_image = infer_image[:,:,1]
B_infer_image = infer_image[:,:,2]

A_infer_image = to_tensor(A_infer_image)
B_infer_image = to_tensor(B_infer_image)

A_infer_image = A_infer_image[None, :, :, :]
B_infer_image = B_infer_image[None, :, :, :]

image = torch.stack ((L_infer_image, A_infer_image, B_infer_image), dim = 2)
image = image [0, 0, :, :, :]
print (image.shape)
image = to_numpy(image)
print( image.shape)
image = np.moveaxis(image, 0, -1)
show_img(image)

exit()

# full_img_predict = np.dstack ((L_infer_image_numpy, np.uint8(255*predict)))
"""

image = torch.stack ((L_infer_image[0, 0, :, :], predict_A, predict_B), dim = 2)
print(image.shape)
# exit()
# image = image [0, 0, :, :, :]
image = to_numpy(image)
image = np.uint8(image*255)
# print (np.uint8(255*predict))

full_img_predict = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
# full_img_predict = cv2.resize (full_img_predict, (w,h))
# print(full_img_predict)
# show_img(L_infer_image_numpy)

show_img(full_img_predict)

# for i in range (0, 10):
#     for j in range (0, 3):
#         print(full_img_predict[45+i, 100+j,:])





