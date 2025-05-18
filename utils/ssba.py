import numpy as np
import bchlib
import tensorflow as tf
from PIL import Image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants



class issbaEncoder(object):
    def __init__(self,model_path, secret,size) :
        BCH_POLYNOMIAL = 137
        BCH_BITS = 5
        self.size = size
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5) 
        # ,config=tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.compat.v1.InteractiveSession(graph=tf.Graph())

        model = tf.compat.v1.saved_model.loader.load(self.sess, [tag_constants.SERVING], model_path)

        input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
        input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
        self.input_secret = tf.compat.v1.get_default_graph().get_tensor_by_name(input_secret_name)
        self.input_image = tf.compat.v1.get_default_graph().get_tensor_by_name(input_image_name)

        output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
        output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
        self.output_stegastamp = tf.compat.v1.get_default_graph().get_tensor_by_name(output_stegastamp_name)
        self.output_residual = tf.compat.v1.get_default_graph().get_tensor_by_name(output_residual_name)
        #bchlib.__version__<1.0.0
        # bch = bchlib.BCH(BCH_POLYNOMIAL,BCH_BITS)
        #bchlib.___version__==1.0.0
        bch = bchlib.BCH(t=BCH_BITS,prim_poly=BCH_POLYNOMIAL)
        if len(secret) > 7:
            print('Error: Can only encode 56bits (7 characters) with ECC')
            return

        data = bytearray(secret + ' '*(7-len(secret)), 'utf-8')
        ecc = bch.encode(data)
        packet = data + ecc

        packet_binary = ''.join(format(x, '08b') for x in packet)
        secret = [int(x) for x in packet_binary]
        secret.extend([0,0,0,0])
        self.secret = secret

    def __call__(self, image):
        image = image.resize((224, 224))
        image = np.array(image, dtype=np.float32) / 255.

        feed_dict = {
            self.input_secret:[self.secret],
            self.input_image:[image]
            }
        hidden_img, residual = self.sess.run([self.output_stegastamp, self.output_residual],feed_dict=feed_dict)
        hidden_img = (hidden_img[0] * 255).astype(np.uint8)
        # residual = residual[0] + .5  # For visualization
        # residual = (residual * 255).astype(np.uint8)
        im = Image.fromarray(np.array(hidden_img))
        # im.save(f'utils/imagenet_ssba/{issbaEncoder.count}_hidden.png')
        return im


    # def output_hidden_img(self,image):
    #     self.i = self.i + 1
    #     img = Image.open(f'utils/imagenet_ssba/{self.i}_hidden.png')
    #     return img
