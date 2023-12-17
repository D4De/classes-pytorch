from classes_core.error_simulator_keras import ErrorSimulator, create_injection_sites_layer_simulator
import classes_core  
'''
def check_classes_layer_compatibility_dev(layer):
        if isinstance(layer,keras.layers.BatchNormalization):
            return "batchnorm"
        elif isinstance(layer,tf.compat.v1.keras.layers.BatchNormalization):
            return "batchnorm"
        elif isinstance(layer,keras.layers.MaxPooling2D):
            return "maxpool"
        elif isinstance(layer,keras.layers.AveragePooling2D):
            return "avgpool"
        elif isinstance(layer,keras.layers.Conv2D):
            return "conv_gemm"
        else:
            return None
'''


class CLASSES_MODELS_PATH:
    models      = classes_core.warp_model_path
    models_warp = classes_core.warp_model_path
    models_warp = classes_core.warp_model_path

print(CLASSES_MODELS_PATH.models)

layer_type = 'conv_gemm'
layer_output_shape_cf = '(None, 16, 27, 27)'
layer_output_shape_cl = '(None, 27, 27, 16)'

shape = (None,16,27,27)
inverted_shape = (shape[0],shape[3],shape[1],shape[2])

available_injection_sites, masks, error_ids = create_injection_sites_layer_simulator(
                                                                            150,
                                                                            layer_type,
                                                                            str(inverted_shape), 
                                                                            str(shape),
                                                                            CLASSES_MODELS_PATH.models_warp,
                                                                            return_id_errors = True,
                                                                            range_min=-30,
                                                                            range_max=+30,
                                                                            #verbose=verbose
                                                                            )
print("GENERATED INJECTIONS")


x = ErrorSimulator(available_injection_sites,masks,len(available_injection_sites),error_ids,name="classes")