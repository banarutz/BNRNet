from imports import *
import os
from torchvision import transforms
from PIL import Image


def generate_paths(dir_list):

    paths = []
    for dir in dir_list:
        for element in os.listdir(dir):
            full_path = os.path.join(dir, element)
            paths.append(full_path)
    return paths

class VGGLoader (torch.utils.data.Dataset):
    def __init__ (self, PATH):
        self.PATH = PATH
        self.paths = os.listdir(PATH)

    def __getitem__ (self, index):
        path = self.paths[index]

        image = cv2.imread(self.PATH + path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        input = image [:, :, 0]
        image = image[:, :, 1:3]
        image = Image.fromarray(np.uint8(image))
        input = Image.fromarray(np.uint8(input))

        transform = transforms.Resize((224, 224), interpolation= transforms.InterpolationMode.BILINEAR)

        image = transform (image)
        input = transform (input)

############  ADD CHECKS FOR DIMENSIONS #################
#         transforms = torch.nn.Sequential(
#     transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)), ### Oare e bine sa fac normalizare si dupa sa transform in LAB?
# )

        to_tensor = transforms.ToTensor()

        image = to_tensor(image)
        input = to_tensor(input)

        # print (image.type, input.type)

        return input, image
        

    def __len__ (self):
        # return len(self.paths)
        return 200


class ValidationDataSet (torch.utils.data.Dataset):
    def __init__(self):

        pass





