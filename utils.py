from imports import *


def compute_PSNR (original_image, transformed_image):
    """
        Compute the PSNR between the original_image and the transformed_image.
    """
    module_diff = np.abs(transformed_image - original_image)
    MSE = np.mean(module_diff ** 2)
    PSNR = 10 * np.log10(np.max(original_image) ** 2 / MSE)
    
    return PSNR


def add_blur (original_image, factor, kernel_size = (3,3), type = 'averaging', sigmaX = 0):
    """
        Add blur to original_image with the strength of the factor specified.
    args:
        original_image: original image;
        factor: takes values between 0 and 1;
        kernel_size: kernel size used for blur. Default value is (3,3);
        type: because of using the cv2 methods, the parameter type can be: 'averaging', 'gaussian', 'median' or 'bilateral';
        sigmaX: is the standard deviation of the gaussian blur on axis X. Default value is 0;
    """
    if type == 'averaging':
        blurry_image = cv2.blur(original_image, kernel_size)
    elif type == 'gaussian':
        blurry_image = cv2.GaussianBlur(original_image, kernel_size ,sigmaX) 
    elif type == 'median':
        blurry_image = cv2.medianBlur(original_image, kernel_size[0])
    elif type == 'bilateral':
        blurry_image = cv2.bilateralFilter(original_image, kernel_size[0], np.floor (factor*100), np.floor(factor*100))
    else:
        raise Exception('Type argument accepted: averaging, gaussian, median or bilateral.')

    return blurry_image

def add_noise (original_image, factor):
    
    if len(original_image.shape) < 3:
        '''In this case, the image is grayscale so only noise we can add is salt and pepper.'''
        s_n_p_matrix = np.random.choice([0,1], size = original_image.shape, p = factor)
        noisy_image = np.logical_and (original_image, s_n_p_matrix)
    elif len(original_image.shape) == 3:
        '''Color image.'''
        # To be added after i receive more information about how to do this.
    return noisy_image

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def show_img (img):
    cv2.imshow ('tzaka paka', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def get_mse (predict, gt):
    pixel = predict.shape[2]*predict.shape[1]
    mse_diff = np.sum((predict - gt) ** 2)/pixel
    return mse_diff	

def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss