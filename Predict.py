#Austin H Kim

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

def loadImg(file):
    img = load_img(file, target_size=(100, 100))
    img = img_to_array(img)
    img = img.reshape(1, 100, 100, 3)
    return img

if __name__ == "__main__":

    # You can try the prediction model by downloading a jpg photo of a cat or a dog into the same
    # directory/folder as this program. Type the name of the file into the file variable below and
    # run the model. Its accuracy is 80%. There are already two .jpg files that you can try out.
    # lion1.jpg and doggo1.jpg

    file = "husky1.jpg"

    img = loadImg(file)

    # The VGG3_Model.h5 is
    model = load_model("VGG3_Model.h5")
    result = model.predict(img)
    if result[0] > 0.5:
        print("It's a dog!!")
    else:
        print("It's a cat!!!")