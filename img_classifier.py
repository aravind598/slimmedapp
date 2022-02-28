import tensorflow as tf
from PIL import Image
import numpy as np
import io


def prepare(bytestr, img_shape=224, rescale=False, expand_dims=False):
    img = tf.io.decode_image(bytestr, channels=3, dtype=tf.dtypes.float32)
    #img = tf.image.resize(img, [img_shape, img_shape])
    if rescale:
        img = img/255.
        img = img.numpy()
    else:
        pass
    if expand_dims:
        return tf.cast(tf.expand_dims(img, axis=0), tf.dtypes.float32)
    else:
        return img.numpy()

"""
def prediction(model, pred):
    prednumpyarray = model.predict(pred)
    print(prednumpyarray.shape)
    predarray = tf.keras.applications.efficientnet.decode_predictions(prednumpyarray, top=5)
    return predarray
"""

def prediction_my(model, pred):
    classes = ["Fruit", "Dog", "Person", "Car", "Motorbike", "Flower", "Cat"]
    # run the inference
    prediction = model.predict(pred)
    #print(classes[prediction.argmax()])
    return classes[prediction.argmax()]

def prepare_my(bytestr: bytes, shape = (1,224,224,3), colab=False):
    """[summary]

    Args:
        bytestr (bytes): [image bytestr from read]
        shape (tuple, optional): [description]. Defaults to (1,224,224,3).

    Returns:
        ndarray: [Output the data in the form of [1,224,224,3] ]]
    """    
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape, dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open(io.BytesIO(bytestr)).convert('RGB')
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    #size = (img_shape, img_shape)
    #image = ImageOps.fit(image, size, Image.ANTIALIAS)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = image_array.astype(np.float32)
    if colab:
        normalized_image_array /= 255.0
    else: normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    
        
    # Load the image into the array
    data[0] = normalized_image_array
    return data



#@st.experimental_singleton # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(model, image):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction. Using the EfficientNet

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    image_array = prepare(image,expand_dims=True)
    #image_pred = prediction(model,image_array)
    # run the inference
    prediction = model.predict(image_array)
    listing = tf.keras.applications.efficientnet.decode_predictions(
        prediction, top=5
    )
    #print(classes[prediction.argmax()])
    return str(listing)
    #return str(image_pred)


def make_my_prediction(my_model,image,colab=False):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction. Using My Model

    Returns:
     image (preproccessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """
    image_array = prepare_my(image)
    #image_pred = prediction_my(my_model,image_array,colab)
    classes = ["Car", "Cat", "Dog", "Flower", "Fruit", "Motorbike", "Person"]
    if colab:
        classes = ['Airplane', 'Bird', 'Car', 'Cat', "Dog",
                   "Flower", "Fruit", "Motorcycle", "Person"]
    # run the inference
    prediction = my_model.predict(image_array)
    #print(classes[prediction.argmax()])
    return str(classes[prediction.argmax()])
    #return str(image_pred)

'''
def prediction_my(model, pred, colab):
    """[Prediction using my model]

    Args:
        model ([type]): [description]
        pred ([type]): [description]

    Returns:
        [type]: [description]
    """
    classes = ["Car","Cat","Dog", "Flower", "Fruit", "Motorbike", "Person"]
    if colab:
        classes = ['Airplane', 'Bird', 'Car', 'Cat', "Dog", "Flower", "Fruit", "Motorcycle", "Person"]
    # run the inference
    prediction = model.predict(pred)
    #print(classes[prediction.argmax()])
    return classes[prediction.argmax()]
'''

def getOutput(interpreter, input_data, input_details=None, output_details=None, colab=False):
    """[Get output from interpreter using tflite models]

    Args:
        interpreter ([type]): [description]
        input_data ([type]): [description]

    Returns:
        [type]: [description]
    """
    #
    if not input_details or not output_details:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    #print(input_details)
    #print(output_details)
    # Test the model on random input data.
    #input_shape = input_details[0]['shape']
    #input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    if colab:
        colab_classes = ['Airplane', 'Bird', 'Car', 'Cat', 'Dog', 'Flower', 'Fruit', 'Motorbike', 'Person']
        return colab_classes[output_data.argmax()]
    else:
        classes = ["Car","Cat","Dog", "Flower", "Fruit", "Motorbike", "Person"]
        return classes[output_data.argmax()]
    
    #print(output_data)

"""
def azure_prediction(jsonImage, uri):
    url = uri.replace("/score","")
    requests.get(url)
    headers = {"Content-Type": "application/json"}
    response = requests.post(uri, data=jsonImage, headers=headers)
    print(response.json())
    return response.json()"""
        





"""
def our_image_classifier(image):
    '''
            Function that takes the path of the image as input and returns the closest predicted label as output
            '''
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)
    # Load the model
    model = tensorflow.keras.models.load_model(
        'model/name_of_the_keras_model.h5')
    # Determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    # Turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (
        image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    labels = {0: "Class 0", 1: "Class 1", 2: "Class 2",
              3: "Class 3", 4: "Class 4", 5: "Class 5"}
    # Run the inference
    predictions = model.predict(data).tolist()
    best_outcome = predictions[0].index(max(predictions[0]))
    print(labels[best_outcome])
    return labels[best_outcome]"""



"""
def prepare_my(bytestr, img_shape=224):
    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Replace this with the path to your image
    image = Image.open(io.BytesIO(bytestr))
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (img_shape, img_shape)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    return data
"""

'''
@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield

def createserver():
    pass
'''


