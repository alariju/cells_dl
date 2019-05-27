import os

import cv2
import numpy as np
from torchvision.transforms import functional
import torch
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms

# import pdb
from cnn import ResNET50

CAM_DIMS = 256

# Activation maps or features extracted are store here
activationMaps = []
"""
This function will be called when a forward is done for the model
It should have the following signature:
hook(module, input, output) -> None
"""


def extract_features(module, input, output):
    activationMaps.append(output.data.cpu().numpy())


def get_weights_fcn():
    # We load resnet18 with pretrained weights
    model = ResNET50()
    model.load_state_dict(torch.load("weights/weights_file"))
    model.eval()

    # ccn_model = models.resnet18(pretrained=True)
    # # the model will be used only for testing
    # ccn_model.eval()

    name_final_conv_layer = 'res_net_50_convolution'
    """
    Registers a forward hook on the module.
    The hook will be called every time after forward() has computed an output. 
    It should have the following signature:
    hook(module, input, output) -> None
    """
    # hook up the function to be called when forward is done
    model._modules.get(name_final_conv_layer).register_forward_hook(
        extract_features)
    # parameters extracts all the parameters calculated for the model
    weights = list(model.parameters())
    # we extract the weights used in the last fully connected layer, used by
    # the final softmax
    weights_fcl = weights[-2].data.numpy();
    # weightsFCL2 = np.squeeze(weightsFCL);
    return weights_fcl, model


"""
Gets the activation maps for classes in classIds
@param activationMaps, input activation maps or feature maps
@param weightsFCN, weights from the fully connected net
@return classesActivationMaps
"""


def get_activation_maps_for_classes(activation_maps, weights_fcn, class_ids):
    # generate the class activation maps upsample to 256x256
    size_to_upsample = (CAM_DIMS, CAM_DIMS)
    bz, number_of_channels_activation_map, height_activation_map, weight_activation_map = activation_maps.shape
    # list of the class activation maps
    output_class_activation_maps = []
    # for each class, compute its class activation map
    for class_id in class_ids:
        # linear combination of the activation maps with the weights
        # W_1,k A_1 + W_2,k A_2 ....
        # use the dot product for this, since weightsFCN has the weights for all the activation maps
        # 512 activation maps for resnet
        print(weights_fcn.shape)
        class_activation_map = weights_fcn[class_id].dot(
            activation_maps.reshape((number_of_channels_activation_map,
                                     height_activation_map * weight_activation_map)))
        class_activation_map = class_activation_map.reshape(
            height_activation_map, weight_activation_map)
        # normalization and scaling
        class_activation_map = class_activation_map - np.min(
            class_activation_map)
        class_activation_map_image = class_activation_map / np.max(
            class_activation_map)
        class_activation_map_image = np.uint8(255 * class_activation_map_image)
        # concat to list of class activation maps
        output_class_activation_maps.append(
            cv2.resize(class_activation_map_image, size_to_upsample))
    return output_class_activation_maps


"""
Use pytorch transforms to preprocess input image
@param image, image to preprocess
@return preprocessedImage
"""


def preprocess_image(input_image):
    input_image = input_image.convert("RGB")
    input_image = functional.adjust_gamma(input_image, gamma=8.)
    cv2_image = np.array(input_image)
    preprocess_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # IMG_URL = 'http://media.mlive.com/news_impact/photo/9933031-large.jpg'
        # response = requests.get(IMG_URL)
        # img_pil = Image.open(io.BytesIO(response.content))
    ])
    preprocessed_image = preprocess_transforms(input_image)
    return preprocessed_image, cv2_image


"""
Predict class
@param cnnModel, cnn model to make the the prediction
@param imageTensor, image tensor to predict its class
@return estimated class id using the model
"""


def predict_class(cnn_model, image_tensor):
    # unsqueeze to eliminate one dimension [1, 3, 224, 224] -> [3, 224, 224]
    image_variable = Variable(image_tensor.unsqueeze(0))
    # raw cnn model prediction
    raw_prediction = cnn_model(image_variable)
    # download the imagenet category list
    # classes = {int(key): value for (key, value)
    #            in requests.get(LABELS_URL).json().items()}
    classes = {1: 'with cells', 0: 'no cells'}
    # calculate softmax output
    softmax_output = F.softmax(raw_prediction, dim=1).data.squeeze()
    # get class probabilities and its corresponding classIds sorted
    class_probabilities, class_ids = softmax_output.sort(0, True)
    class_probabilities = class_probabilities.numpy()
    class_ids = class_ids.numpy()

    # output the prediction
    for i in range(2):
        print('{:.3f} -> {}'.format(class_probabilities[i],
                                    classes[class_ids[i]]))
    # Return the id of the best class
    return class_ids[0]


"""
Draws and writes the CAM
@param originalImageName, image used to draw heat overlay
@param camImageName, name of image to write
@param activationMapsClass, 
"""


def write_class_activation_maps(image_cv, cam_image_name,
                                activation_maps_class):
    # render the CAM and output
    # print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])
    # image_cv = cv2.imread(original_image_name)
    height, width, _ = image_cv.shape
    heatmap = cv2.applyColorMap(
        cv2.resize(activation_maps_class[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + image_cv * 0.5
    cv2.imwrite(cam_image_name, result)


def get_random_evaluation_image_path_from_folder(folder):
    for root, _, files in os.walk(folder):
        np.random.shuffle(files)
        return files[0], os.path.join(root, files[0])


def format_image_name(image_name):
    return image_name.replace(".tif", ".jpg")


def am_generate():
    print("Load model")
    weights_fcn, cnn_model = get_weights_fcn()
    print("Opening image...")
    pil_image_name, pil_image_path = \
        get_random_evaluation_image_path_from_folder(
            "datasets/cells/train/with_cells")
    pil_image = Image.open(pil_image_path)
    # imagePIL.save("test.jpg")
    image_tensor, cv2_image = preprocess_image(pil_image)
    # unsqueeze adds one dimension (empty)
    print("Predicting class...")
    class_id = predict_class(cnn_model, image_tensor)
    print(class_id)
    print("Calculating activation maps for class...")
    # calculate the activation maps for only the best class, first activation
    # map obtained
    activation_maps_top_class = get_activation_maps_for_classes(
        activationMaps[0], weights_fcn, [class_id])
    print(pil_image_path)
    am_file_name = f'activation_map-{format_image_name(pil_image_name)}'
    write_class_activation_maps(cv2_image, am_file_name,
                                activation_maps_top_class)


if __name__ == '__main__':
    am_generate()
