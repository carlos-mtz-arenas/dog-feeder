import torch
from PIL import Image
from glob import glob
from os import path

from localml.torch_components import DogModel, generate_transformer

# XXX: fix the package reference to avoid doing this
current_path = __file__.replace('/test_utils.py', '')


def get_model():
    device = torch.device("cpu")
    # import our own model already configured
    model = DogModel(num_classes=2).to(device)

    model_checkpoint = torch.load(current_path + '/model_checkpoint.pt')

    # extract the classes
    classes = model_checkpoint["classes"]

    # load the already trained state into the model instance
    model.load_state_dict(model_checkpoint["state"])
    model.eval()

    return (model, classes)


def get_image(path):
    pil_img = Image.open(path)
    transformer = generate_transformer()
    tensor_img = transformer(pil_img)
    tensor_img = tensor_img.float().unsqueeze(0)

    return tensor_img


def predict_class_for_image(model, classes, img):
    input_tensor = get_image(img)
    output = model(input_tensor)
    _, pred = torch.max(output, 1)
    return {"class": classes[pred], "prediction": pred}


# only for "unit testing" the model
if __name__ == '__main__':
    (model, classes) = get_model()
    for img in glob(current_path + '/quick_test/*.jpg'):
        predict = predict_class_for_image(model, classes, img)
        print('Input: [{}]l Prediction: [{}]'
              .format(img, predict['class']))
