from typing import Optional
import tensorflow as tf
import numpy as np
from .injection_sites_generator import (
    InjectableSite,
    InjectionSitesGenerator,
    InjectionValue,
)
from enum import IntEnum
import sys
from tqdm import tqdm

from .loggers import get_logger

logger = get_logger("ErrorSimulator")


def create_injection_sites_layer_simulator(
    num_requested_injection_sites: int,
    layer_type: str,
    layer_output_shape_cf: str,
    layer_output_shape_cl: str,
    models_folder: str,
    range_min: float = None,
    range_max: float = None,
    fixed_spatial_class: Optional[str] = None,
    fixed_domain_class: Optional[dict] = None,
    return_id_errors=False,
    verbose=False,
):

    def __generate_injection_sites(sites_count: int, layer_type: str, size: int):

        injection_site = InjectableSite(layer_type, size)

        injection_sites = InjectionSitesGenerator(
            [injection_site], models_folder, fixed_spatial_class, fixed_domain_class
        ).generate_random_injection_sites(sites_count)

        return injection_sites

    if range_min is None or range_max is None:
        range_min = -30.0
        range_max = 30.0
        logger.warn(
            f"""range_min and/or range_max are not specified, so their default values will be kept ({range_min}, {range_max}). 
                        You may want to change the defaults by calculating the average range_min and range_max average, 
                        and specify them as inputs of this function. """
        )

    available_injection_sites = []
    masks = []
    patterns = []

    ids = []  # FOR_NIC_REPORTS (list of error patterns)

    for _ in tqdm(range(num_requested_injection_sites), disable=not verbose):
        ids.append(
            (0, 0)
        )  # FOR_NIC_REPORTS (list of error patterns) => ASK IF IT IS POSSIBLE TO EXTRACT PATTERNS USED
        curr_injection_sites = __generate_injection_sites(
            1, layer_type, layer_output_shape_cf
        )
        pattern = curr_injection_sites[0].get_pattern()
        shape = eval(layer_output_shape_cl.replace("None", "1"))
        curr_inj_nump = np.zeros(shape=shape[1:])
        curr_mask = np.ones(shape=shape[1:])

        if len(curr_injection_sites) > 0:
            for idx, value in curr_injection_sites[0].get_indexes_values():
                channel_last_idx = (idx[0], idx[2], idx[3], idx[1])
                curr_mask[channel_last_idx[1:]] = 0
                curr_inj_nump[channel_last_idx[1:]] += value.get_value(
                    range_min, range_max
                )

            available_injection_sites.append(curr_inj_nump)
            masks.append(curr_mask)
            patterns.append(pattern)

    if return_id_errors:
        return available_injection_sites, masks, patterns
    else:
        return available_injection_sites, masks


class ErrorSimulatorMode(IntEnum):
    disabled = (1,)
    enabled = 2


@tf.function
def fault_injection_batch_v2(
    inputs, __num_inj_sites, __available_injection_sites, __masks, error_simulator
):
    shape = tf.shape(inputs)
    batch_size = shape[0]

    random_indexes = tf.random.uniform(
        shape=[batch_size], minval=0, maxval=__num_inj_sites, dtype=tf.int32, seed=22
    )

    random_tensor = tf.gather(__available_injection_sites, random_indexes)
    random_mask = tf.gather(__masks, random_indexes)

    random_indexes = tf.expand_dims(random_indexes, axis=1)

    error_simulator.history.assign(
        tf.concat([error_simulator.history, random_indexes], axis=0)
    )
    # tf.print(random_tensor.shape,random_mask.shape)

    return (inputs * random_mask + random_tensor), tf.zeros_like(random_mask)  # WARNING


class ErrorSimulator(tf.keras.layers.Layer):

    def __init__(
        self, available_injection_sites, masks, num_inj_sites, error_ids, **kwargs
    ):

        super(ErrorSimulator, self).__init__(**kwargs)
        self.__num_inj_sites = num_inj_sites
        self.__available_injection_sites = []
        self.__masks = []
        self.error_ids = error_ids
        self.history = tf.Variable(
            initial_value=[[1]],
            dtype=tf.int32,
            shape=[None, 1],
            trainable=False,
            name="history",
        )

        # Parameter to chose between enable/disable faults
        self.mode = tf.Variable(
            [[int(ErrorSimulatorMode.enabled)]],
            shape=tf.TensorShape((1, 1)),
            trainable=False,
            name="mode",
        )

        for inj_site in available_injection_sites:
            self.__available_injection_sites.append(
                tf.convert_to_tensor(inj_site, dtype=tf.float32)
            )
        for mask in masks:
            self.__masks.append(tf.convert_to_tensor(mask, dtype=tf.float32))

    """
    Allow to enable or disable the Fault Layer
    """

    def set_mode(self, mode: ErrorSimulatorMode):
        self.mode.assign([[int(mode)]])

    # GET HISTORY OF USED ERROR PATTERN
    def get_history(self):
        return self.history.numpy()

    # CLEAR HISTORY OF USED ERROR PATTERN
    def clear_history(self):
        self.history.assign(
            tf.Variable(
                initial_value=[[1]], dtype=tf.int32, shape=[None, 1], trainable=False
            )
        )

    def get_config(self):
        config = super().get_config()
        return config

    @tf.custom_gradient  # NIC_FOR_FAT_PURPOSES
    def call(self, inputs):
        # tf.print("MODE LAYER :", self.mode, tf.constant([[int(ErrorSimulatorMode.disabled)]]), output_stream=sys.stdout)
        # TF operator to check which mode is active
        # If Disabled => Return Vanilla output
        # If Enabled  => Return Faulty  output

        a = tf.cond(
            self.mode == tf.constant([[int(ErrorSimulatorMode.disabled)]]),
            true_fn=lambda: (inputs, tf.ones_like(inputs)),
            false_fn=lambda: fault_injection_batch_v2(
                inputs,
                self.__num_inj_sites,
                self.__available_injection_sites,
                self.__masks,
                self,
            ),
        )

        # NIC_FOR_CUSTOM_GRADIENT

        def grad(upstream):
            return tf.multiply(upstream, a[1])

        """
        def grad(upstream):
            return upstream
        """
        return a[0], grad

        """

        #NIC_FOR_CUSTOM_GRADIENT
        def grad(upstream):
            mask = tf.cond(self.mode == tf.constant([[int(ErrorSimulatorMode.disabled)]]), 
                       true_fn=lambda: tf.ones_like(inputs),
                       false_fn=lambda: tf.zeros_like(inputs))
            
            return  tf.multiply(mask,upstream) 
        """
