import os
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.figure import Figure
plt.switch_backend('agg')

class Eye_explain:
    def __init__(self, my_model):
        self.layer_name = "conv2d_15"
        self.visualization_model = tf.keras.models.Model(inputs=my_model.input,
                                                    outputs=my_model.get_layer(self.layer_name).output)

    def explain(self, image,eye_type, encounter_number):
        folder_name = 'eye_cataract_model/images/' + encounter_number
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        activations = self.visualization_model.predict(image)
        fig = Figure(figsize=(80, 80))
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)
        if eye_type==1:
            eye='Right'
        else:
            eye='Left'
        for i in range(activations.shape[-1]):
            plt.subplot(4, 6, i + 1)
            plt.imshow(activations[0, :, :, i], )
            plt.axis('off')
        file_name= folder_name +'/'+ encounter_number+'_'+eye + '.jpg'
        plt.savefig(file_name, dpi=1200)
        plt.close()
        return file_name
