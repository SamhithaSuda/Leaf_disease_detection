import os
import argparse
import subprocess
import numpy as np
import cv2

from PIL import Image

from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.inception_v3 import preprocess_input






CORN = 'corn'
SPECIES = [CORN]
DISEASE_SUPPORTED_SPECIES = {CORN }


CORN_CLASSES = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy']


# all species classes with their species name as key
PLANT_CLASSES = {  
    CORN: CORN_CLASSES
}

# types of models to be used for predictions
VGG_ARCHITECTURE = 'vgg'
INCEPTIONV3_ARCHITECTURE = 'inceptionv3'
SUPPORTED_MODEL_TYPES = {VGG_ARCHITECTURE, INCEPTIONV3_ARCHITECTURE}

# modes of detection i.e detecting plant disease or species
DISEASE_DETECTION = 'disease_detection'


# image target sizes for our supported model architectures
TARGET_IMAGE_SIZES = {
    VGG_ARCHITECTURE: {
        DISEASE_DETECTION: (64, 64)
    },
    INCEPTIONV3_ARCHITECTURE: {
        DISEASE_DETECTION: (100, 100)
    }
}

VGG_MODELS = {
    CORN: 'Corn_0.8926_VGG.h5'
}

INCEPTIONV3_MODELS = {
    CORN: 'InceptionV3-scratch_segCorn.h5'
}

# base path from where models will be loaded
MODEL_STORAGE_BASE = 'Models'


def get_classes(species_name):
    return PLANT_CLASSES[species_name]


def get_disease_model(species, model_type):

    if species not in DISEASE_SUPPORTED_SPECIES:
        raise ValueError("`{}` species has no disease model yet.\n"
                         "Species tha have disease models are {}".format(species, DISEASE_SUPPORTED_SPECIES))

    if model_type == VGG_ARCHITECTURE:
        return VGG_MODELS[species]
    elif model_type == INCEPTIONV3_ARCHITECTURE:
        return INCEPTIONV3_MODELS[species]
    else:
        raise ValueError("No such `{}` model type is supported.\n"
                         "Supported model types are {}".format(model_type, SUPPORTED_MODEL_TYPES))



def get_predictions(model_path, img_path, img_target_size):
    if not os.path.exists(model_path):
        raise ValueError('No such `{}` file found\n'
                         'Please, checkout the readme of the project '
                         'on github and download required models'.format(model_path))
    model = load_model(model_path)

    # get image as array and resize it if necessary
    pil_img = Image.open(img_path)
    if pil_img.size != img_target_size:
        pil_img = pil_img.resize(img_target_size)

    img = image.img_to_array(pil_img)

    # if alpha channel found, discard it
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # preprocess image
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    preds = model.predict(img).flatten()

    # get predictions index sorted based on the best predictions
    value_ = preds.argsort()
    sorted_preds_index = value_[::-1]

    return preds, sorted_preds_index


def segment_image(img_path):

    image_name, extension = os.path.splitext(img_path)
    segmented_image_name = image_name + "_marked" + extension  
    result = subprocess.check_output(['python', "leaf-image-segmentation/segment.py", "-s", img_path])
    print('Info: Input image segmented.')

    return segmented_image_name



def predict_disease(img_path, species, model_type=VGG_ARCHITECTURE, do_print=True):

    if species not in SPECIES:
        raise ValueError("No such `{}` species is supported.\n"
                         "Supported species are {}".format(species, SPECIES))

    if species not in DISEASE_SUPPORTED_SPECIES:
        print("Info: For `{}` species, a disease can not be predicted "
              "since its disease model is not implemented yet.".format(species))
        return None
    else:
        SPECIES_CLASSES = get_classes(species)
        model_path = os.path.join(MODEL_STORAGE_BASE, get_disease_model(species, model_type))

        target_image_size = TARGET_IMAGE_SIZES[model_type][DISEASE_DETECTION]
        preds, sorted_preds_index = get_predictions(model_path, img_path, target_image_size)

        if do_print:
            print("Plant Disease : ")
            for i in sorted_preds_index:
                print("\t-" + str(SPECIES_CLASSES[i]) + ": \t" + str(preds[i]))

        return str(SPECIES_CLASSES[sorted_preds_index[0]])


def get_cmd_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help='Image file path')
    args = parser.parse_args()

    return args


if __name__ == "__main__":

        args = get_cmd_args()
        image_args = cv2.imread(args.image)
        kernel = np.ones((7,7),np.float32)/25
        img1 = cv2.filter2D(image_args,-1,kernel)
        cv2.namedWindow('Gaussian Filter',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Gaussian Filter', 300,200)
        cv2.imshow('Gaussian Filter', img1)
        img3 = cv2.bilateralFilter(img1,9,75,75)
        cv2.namedWindow('Bilateral Filtered Image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Bilateral Filtered Image', 300,200)
        cv2.imshow('Bilateral Filtered Image',img3)
        segmentimg=segment_image(args.image)
        image_segment = cv2.imread(segmentimg)
        cv2.namedWindow('Segmented Image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Segmented Image',300,200)
        cv2.imshow('Segmented Image',image_segment)
        species="corn"
        print("---------------------", species)
        diseasereturn=predict_disease(segmentimg,  species)
        print("=================================")
        print(diseasereturn)
        print("================================")
        # height, width, depth = image_args.shape
        # desired_height = 512
        # aspect_ratio = desired_height/width
        # dimension = (desired_height, int(height*aspect_ratio))
        # img_resized = cv2.resize(image_args, dimension)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # org = (10,650)
        # fontScale = 2
        # color = (255, 255, 255)
        # thickness = 2
        # cv2.putText(image_args, diseasereturn, org, font,  fontScale, color, thickness, cv2.LINE_AA)
        # cv2.namedWindow('Final Ouput',cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Final Ouput',600,400)
        # cv2.imshow('Final Ouput',img_resized)
        if diseasereturn == "Corn_(maize)___Northern_Leaf_Blight":
        	print("******************************************************************")
        	print("===================================================================")
        	print(" NORTERN LEAF BLIGHT DISEASE DETECTED -- APPROXIMATE REASONS")
        	print("===================================================================")
        	print("1 - FUNGUS ")
        	print("2 - BEING WET FOR LONG HOURS")
        	print("///////////////////////////////////////////////////////////////////")
        	print(" PRECAUTIONS : It Does not stay wet for long periods of time")
        	print("///////////////////////////////////////////////////////////////////")
        	print(" -------------- PRESS 'q'  to Quit the Program --------------------")
        	print("*******************************************************************")
        if diseasereturn == "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":
        	print("******************************************************************")
        	print("===================================================================")
        	print(" Cercospora_leaf_spot Gray_leaf_spot DETECTED -- APPROXIMATE REASONS")
        	print("===================================================================")
        	print("1 - FUNGUS CERCOSPORA")
        	print("2 - ZEAE-MAYDIS")
        	print("///////////////////////////////////////////////////////////////////")
        	print(" PRECAUTIONS : REGULARY USE PETISICED FROM REOCCURING ")
        	print("///////////////////////////////////////////////////////////////////")
        	print(" -------------- PRESS 'q'  to Quit the Program --------------------")
        	print("*******************************************************************")
        if diseasereturn == "Corn_(maize)___Common_rust_":
        	print("******************************************************************")
        	print("===================================================================")
        	print(" Corn_(maize)___Common_rust_DETECTED -- APPROXIMATE REASONS")
        	print("===================================================================")
        	print("1 -  PUCCININA SORGHI")
        	print("2 -  PATHOGEN")
        	print("3 -  COOL CONDITIONS")
        	print("///////////////////////////////////////////////////////////////////")
        	print(" PRECAUTIONS : MAINTAIN HOT AND DRY CONDITIONS  ")
        	print("///////////////////////////////////////////////////////////////////")
        	print(" -------------- PRESS 'q'  to Quit the Program --------------------")
        	print("*******************************************************************")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
