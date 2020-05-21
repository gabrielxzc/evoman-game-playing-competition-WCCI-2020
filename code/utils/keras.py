from keras import models
from keras import activations
from keras import layers

import numpy as np


def get_model(input_dim, hidden_and_output_layers_sizes, weights=None):
    model = models.Sequential()

    for hidden_or_output_layer_idx, hidden_or_output_layer_nr_of_neurons in enumerate(hidden_and_output_layers_sizes):
        layer_kwargs = {"activation": activations.tanh}

        is_first_hidden_layer = hidden_or_output_layer_idx == 0
        if is_first_hidden_layer:
            layer_kwargs["input_dim"] = input_dim

        model.add(layers.Dense(hidden_or_output_layer_nr_of_neurons, **layer_kwargs))

    if weights is not None:
        model.set_weights(weights)

    return model


def get_model_flattened_weights(model):
    return np.concatenate([layer_weights.flatten() for layer_weights in model.get_weights()])


def get_model_weights_from_flattened_weights(flattened_weights, model_layers_sizes):
    flattened_weights_idx = 0

    nr_of_weights_on_each_layer = get_model_layers_nr_of_weights(model_layers_sizes)
    nr_of_biases_on_each_layer = model_layers_sizes[1:]

    weights = []

    for nr_of_weights, nr_of_biases, previous_layer_size in zip(nr_of_weights_on_each_layer,
                                                                nr_of_biases_on_each_layer,
                                                                model_layers_sizes[:-1]):
        nr_of_neurons_on_layer = nr_of_biases

        layer_weights = np.array(flattened_weights[flattened_weights_idx:flattened_weights_idx + nr_of_weights]) \
            .reshape((previous_layer_size, nr_of_neurons_on_layer))
        flattened_weights_idx += nr_of_weights

        layer_biases = np.array(flattened_weights[flattened_weights_idx:flattened_weights_idx + nr_of_biases])
        flattened_weights_idx += nr_of_biases

        weights.append(layer_weights)
        weights.append(layer_biases)

    return weights


def get_model_layers_nr_of_weights(model_layers_sizes):
    nr_of_weights_on_each_layer = []

    for previous_layer_size, layer_size in zip(model_layers_sizes[:-1], model_layers_sizes[1:]):
        nr_of_weights_on_each_layer.append(previous_layer_size * layer_size)

    return nr_of_weights_on_each_layer
