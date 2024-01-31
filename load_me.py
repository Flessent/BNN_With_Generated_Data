from bnn_to_cnf import *
from bnn import *
import larq as larq




model = BNN(num_neuron_in_hidden_dense_layer=18, num_neuron_output_layer=16, input_dim=18, output_dim=16)
trained_weights ="weights_after_training.h5"
model.load_weights(trained_weights)
print('Weights loaded successfully !!! ')

print('Encoding Process BNN----> CNF started')
dismacs_from_loaded_weights=encode_network(model)
