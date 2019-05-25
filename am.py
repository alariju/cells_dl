# Implementation of CAMs, based on https://github.com/metalbubble/CAM/blob/master/pytorch_CAM.py

# pip install opencv-python

# import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
# import pdb

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
    ccn_model = models.resnet18(pretrained=True)
    # the model will be used only for testing
    ccn_model.eval();
    name_final_conv_layer = 'layer4'
    """
    Registers a forward hook on the module.
    The hook will be called every time after forward() has computed an output. 
    It should have the following signature:
    hook(module, input, output) -> None
    """
    # hook up the function to be called when forward is done
    ccn_model._modules.get(name_final_conv_layer).register_forward_hook(extract_features)
    # paramaters extracts all the parameters calculated for the model
    weights = list(ccn_model.parameters())
    # we extract the weights used in the last fully connected layer, used by the final softmax
    weights_fcl = weights[-2].data.numpy();
    # weightsFCL2 = np.squeeze(weightsFCL);
    return weights_fcl, ccn_model


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
    for classId in class_ids:
        # linear combination of the activation maps with the weights
        # W_1,k A_1 + W_2,k A_2 ....
        # use the dot product for this, since weightsFCN has the weights for all the activation maps
        # 512 activation maps for resnet
        print(weights_fcn.shape)
        classActivationMap = weights_fcn[classId].dot(
            activation_maps.reshape((number_of_channels_activation_map, height_activation_map * weight_activation_map)))
        classActivationMap = classActivationMap.reshape(height_activation_map, weight_activation_map)
        # normalization and scaling
        classActivationMap = classActivationMap - np.min(classActivationMap)
        classActivationMapImage = classActivationMap / np.max(classActivationMap)
        classActivationMapImage = np.uint8(255 * classActivationMapImage)
        # concat to list of class activation maps
        output_class_activation_maps.append(cv2.resize(classActivationMapImage, size_to_upsample))
    return output_class_activation_maps


"""
Use pytorch transforms to preprocess input image
@param image, image to preprocess
@return preprocessedImage
"""


def preprocessImage(image):
    # imagenet mean and std normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocessTransforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    preprocessedImage = preprocessTransforms(image);
    return preprocessedImage;


"""
Predict class
@param cnnModel, cnn model to make the the prediction
@param imageTensor, image tensor to predict its class
@return estimated class id using the model
"""


def predictClass(cnnModel, imageTensor):
    # unsqueeze to eliminate one dimension [1, 3, 224, 224] -> [3, 224, 224]
    imageVariable = Variable(imageTensor.unsqueeze(0))
    # raw cnn model prediction
    rawPrediction = cnnModel(imageVariable)
    # download the imagenet category list
    classes = {int(key): value for (key, value)
               in requests.get(LABELS_URL).json().items()}
    # calculate softmax output
    softmaxOutput = F.softmax(rawPrediction, dim=1).data.squeeze()
    # get class probabilities and its corresponding classIds sorted
    classProbabilities, classIds = softmaxOutput.sort(0, True)
    classProbabilities = classProbabilities.numpy()
    classIds = classIds.numpy()

    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(classProbabilities[i], classes[classIds[i]]))
    # Return the id of the best class
    return classIds[0];


"""
Draws and writes the CAM
@param originalImageName, image used to draw heat overlay
@param camImageName, name of image to write
@param activationMapsClass, 
"""


def writeClassActivationMap(originalImageName, camImageName, activationMapsClass):
    # render the CAM and output
    # print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0]])
    imageCV = cv2.imread(originalImageName)
    height, width, _ = imageCV.shape
    heatmap = cv2.applyColorMap(cv2.resize(activationMapsClass[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + imageCV * 0.5
    cv2.imwrite(camImageName, result)


if __name__ == '__main__':
    LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
    print("Load model")
    (weightsFCN, cnnModel) = get_weights_fcn()
    # IMG_URL = 'http://media.mlive.com/news_impact/photo/9933031-large.jpg'
    # response = requests.get(IMG_URL)
    # img_pil = Image.open(io.BytesIO(response.content))
    print("Opening image...")
    imagePIL = Image.open("demoImage.jpg")
    # imagePIL.save("test.jpg")
    imageTensor = preprocessImage(imagePIL)
    # unsqueeze adds one dimension (empty)
    print("Predicting class...")
    classId = predictClass(cnnModel, imageTensor)
    print(classId)
    print("Calculating activation maps for class...")
    # calculate the activation maps for only the best class, first activation map obtained
    activationMapsTopClass = get_activation_maps_for_classes(activationMaps[0], weightsFCN, [classId])
    writeClassActivationMap("demoImage.jpg", "CAM.jpg", activationMapsTopClass)
