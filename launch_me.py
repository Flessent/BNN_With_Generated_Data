from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from larq.layers import QuantDense
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import accuracy_score
from pysdd.sdd import SddManager
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential
import larq as larq
from bnn import *
from cnf import *
from bnn_to_cnf import*
import math
from  pysat.solvers import Glucose3
import itertools
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import  optimizers
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import Layer

def read_and_print_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                print(line.strip())  # Strip to remove trailing newline characters
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
def get_num_variables_and_clauses_from_cnf(cnf_filename):
    num_variables, num_clauses = None, None
    
    with open(cnf_filename, 'r') as cnf_file:
        for line in cnf_file:
           
            if line.startswith('p cnf'):
               
                _, _, num_variables, num_clauses = line.split()
                num_variables, num_clauses = int(num_variables), int(num_clauses)
                break  
    print('Num Vars :', num_variables)
    print('num clauses :', num_clauses)
                
    return num_variables, num_clauses
def replace_question_mark(sequence):
    return [1 if bit == '?' else int(bit) for bit in sequence]
def write_output(cnf_array, num_terms, filename):
    with open(filename, 'w') as f:
        f.write('p cnf %d %d\n' % (num_terms, len(cnf_array)))
        for clause in cnf_array:
            f.write(' '.join(map(str, clause)) + ' 0\n')
    print("Wrote cnf to %s" % filename)

if __name__ == "__main__":
          
     datafile = 'C:\\Users\\freun\\Desktop\\WS2\\Masterarbeit\\from_Scratch\\training\\Other_Test\\new_cnf_encoding_with_generated_data\\generated_dataset.txt'
     data = np.loadtxt(datafile, dtype=str)  

     X = data[:, 0] 
     Y = data[:, 1] 

     X = np.core.defchararray.replace(X, '?', '1')

     df = pd.DataFrame({'X': X, 'Y': Y})

     X = np.array([list(map(int, binary_string)) for binary_string in df['X']])

     y = np.array([int(label, 2) for label in df['Y']])

     y_one_hot = to_categorical(y, num_classes=16)
     print('X :', X[:10], X.shape)
     print('Y encoded :', y_one_hot[:10], y_one_hot.shape)


     X_train, X_test, Y_train, Y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

     print('X_train :', X_train[:10])
     print('Y_train :',Y_train[:10])

     print("X_train shape:", X_train.shape)
     print("Y_train shape:", Y_train.shape)
     print('Unique :', len(np.unique(Y_train,axis=0)))

     opt=larq.optimizers.Bop(threshold=1e-08, gamma=0.001, name="Bop")
     model = BNN(num_neuron_in_hidden_dense_layer=18, num_neuron_output_layer=len(np.unique(Y_train,axis=0)), input_dim=18, output_dim=len(np.unique(Y_train,axis=0)))

     model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
     initial_weights = model.get_weights()
     model.save_weights("initial_weights.h5")
     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
     #Y_train_encoded = np.argmax(Y_train, axis=1)
     #Y_test_encoded = np.argmax(Y_test_encoded, axis=1)
     

     history = model.fit(X_train, Y_train, epochs=30, batch_size=32, validation_data=(X_test, Y_test), callbacks=[early_stopping])


     test_loss, test_accuracy = model.evaluate(X_test, Y_test)
     print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
     predictions = model.predict(X_test)

     # Convert the predictions to binary values (assuming a threshold of 0.5)
     binary_predictions = (predictions >= 0.5).astype(int)
     precision = precision_score(Y_test, binary_predictions, average='micro')
     recall = recall_score(Y_test, binary_predictions, average='micro')
     f1 = f1_score(Y_test, binary_predictions, average='micro')
     print(f'Precision: {precision:.4f}')
     print(f'Recall: {recall:.4f}')
     print(f'F1 Score: {f1:.4f}')
     correct_predictions = np.sum(np.all(binary_predictions == Y_test, axis=1))
     total_samples = len(Y_test)
     print(f'Correct Predictions: {correct_predictions} out of {total_samples}')

     # Display the predictions
     print("Predictions:")
     print(binary_predictions)
     print('True values')
     print(Y_test)
     print('X_train')
     print(X_train[:10])
     cnf_clauses = Cnf([])
     layer_sizes = [18, 18, 16]
     input_size = layer_sizes[0]
     hidden_size = layer_sizes[1]
     output_size = layer_sizes[2]
     dim = [
    (hidden_size,),  # First QuantDense layer
    (hidden_size,),  # Second QuantDense layer
    (output_size,)  # Output layer with softmax activation
    ]

     print('################################################### Encoding Process starts... ###########################################################################')

     
     input_terms = [Term(annotation='in' + str(i)) for i in range(dim[0][0])]
     output_terms = input_terms

     for layer, layer_id in zip(model.layers, range(len(model.layers))):
        if layer_id < len(model.layers) - 1  and isinstance(layer, QuantDense) :
            #x = layer.get_input()
            weights_in = layer.get_weights()[0]  # Assuming the weights are the first element in the list returned by get_weights
            biases_in = layer.get_weights()[1]
            #print('layer.get_weights()[0]:', weights[0])
            #print('layer.get_weights()[1]:', biases[:5])
            print('Internal :', layer.name)


            output_terms, output_clauses = internal_layer_to_cnf(output_terms, weights_in, biases_in, 'dense_layer_' + str(layer_id))
        else:
            input_terms = [Term(annotation='out' + str(i)) for i in range(dim[-1][0])]
            output_terms = input_terms
            weights_out = layer.get_weights()[0]  # Assuming the weights are the first element in the list returned by get_weights
            biases_out = layer.get_weights()[1]
            print('External :', layer.name)
            output_terms, output_clauses = output_layer_to_cnf(output_terms, weights_out, biases_out, 'dense_layer_' + str(layer_id))

        cnf_clauses += output_clauses
        print(len(output_clauses.clauses))

     s = set()
     d = {}
     for clause in cnf_clauses.clauses:
        for term in clause.terms:
            s.add(abs(term.tid))

     sorted_s = sorted(s)
     for i, tid in enumerate(sorted_s):
        d[tid] = i + 1

     cnf_array = []
     for clause in cnf_clauses.clauses:
        clause_array = []
        for term in clause.terms:
            clause_array.append(d[abs(term.tid)] * sign(term.tid))

        cnf_array.append(clause_array)
     print(len(cnf_array))

     swap0 = dim[0][0] + 1
     swap1 = d[abs(output_terms[0].tid)]
     for i, clause in enumerate(cnf_array):
        for j, term in enumerate(clause):
            if abs(term) == swap0:
                cnf_array[i][j] = swap1 * sign(term)
            elif abs(term) == swap1:
                cnf_array[i][j] = swap0 * sign(term)

     print("swapped %d with %d" % (swap0, swap1))
     output_file='bnntocnf.cnf'
     write_output(cnf_array, len(s), output_file)
     print('################################################### End of the Encoding Process !!!  ###########################################################################')


     

     model.save("model.h5")
     datafile = 'weights_after_training.h5'

     model.save_weights(datafile)
     lq.models.summary(model)
     describe_network(model)
     plt.figure(figsize=(12, 6)) 

    # Plot training & validation accuracy values
     """
     plt.subplot(1, 2, 1)
     plt.plot(history.history['accuracy'])
     plt.plot(history.history['val_accuracy'])
     plt.title('Model accuracy')
     plt.ylabel('Accuracy')
     plt.xlabel('Epoch')
     plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
     plt.subplot(1, 2, 2)
     plt.plot(history.history['loss'])
     plt.plot(history.history['val_loss'])
     plt.title('Model loss')
     plt.ylabel('Loss')
     plt.xlabel('Epoch')
     plt.legend(['Train', 'Validation'], loc='upper left')

     plt.tight_layout()
     plt.show()"""