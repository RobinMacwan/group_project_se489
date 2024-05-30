import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    """
    A class used to manage the prediction process.
    """
    def __init__(self,filename):
        """
        Initialization of the PredictionPipeline class.

        Parameters
        ----------
        filename : str
            The name of the file to be predicted.
        
        """
        self.filename =filename

    def predict(self):
        """
        Method to predict the image.

        This method is responsible for loading the model 
        and predicting the image. Image is classified as
        Tumor or Normal.

        Returns
        -------
        list
            A list containing the prediction result of the image.

        """
        # load model
        #model = load_model(os.path.join("data","training", "model.h5"))
        model = load_model(os.path.join("models", "model.h5"))

        # load and preprocess the image
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = 'Tumor'
            return [{ "image" : prediction}]
        else:
            prediction = 'Normal'
            return [{ "image" : prediction}]