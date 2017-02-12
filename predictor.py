import time
import data_tools
import numpy as np
import theano
import theano.tensor as T
import lasagne
from math import floor
import random
from os import listdir
from os.path import isfile, join

# Parameters
train_batch_size = 2000
test_batch_size = 1000
num_epochs = 1500
several_input_files = False
smallest_input_file_size = 3000
extended_peptide_representation = False
importance_alphabet = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W",
                       "Y", "X"]
# importance_alphabet = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
#training_file = "/home/daniel/Codes/haydars_code/Data/saras_full_train.txt"
training_file = "C:\Users\Daniel\Desktop\data\saras_full_train.txt"

# test_file = "/home/daniel/Codes/haydars_code/Data/saras_test.txt"
#test_file = "/home/daniel/Codes/haydars_code/Data/fdr0001_data/rtime_fdr0.001/Adult_CD8Tcells_Gel_Elite_44_f06.rtimes.tsv"
test_file = "C:\Users\Daniel\Desktop\data\saras_test.txt"
override_test_file = False
max_file_number = 0
# Max_large_files was used in order to only use some part of the files when training with multiple files.
max_large_files = 369 - floor(369 * 0.98)


# Returns a list of files in directory
def get_file_list(mypath):
    file_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return file_list


# Returns number of unique peptides from files in a given directory
def count_unique_peptides(directory):
    file_list = get_file_list(directory)
    peptide_dictionary = dict()
    large_file_peptide_dict = dict()
    peptide_count = 0
    peptide_count_large_files = 0
    largest_file = 0
    largest_file_name = ""
    num_large_files = 0
    for f in range(len(file_list)):
        peptides = data_tools.read_data(directory + file_list[f])
        if len(peptides) > largest_file:
            largest_file = len(peptides)
            largest_file_name = file_list[f]
        if len(peptides) > smallest_input_file_size:
            num_large_files += 1
            for p in range(len(peptides)):
                peptide = peptides[p].sequence
                if peptide not in large_file_peptide_dict:
                    large_file_peptide_dict[peptide] = 1
                    peptide_count_large_files += 1
            if num_large_files == max_large_files:
                max_file_number = f
                print("Max file number: " + str(f))
                print("Number of files used: " + str(max_large_files))
                print("Peptide count in used files: " + str(peptide_count_large_files))
        for p in range(len(peptides)):
            peptide = peptides[p].sequence
            if peptide not in peptide_dictionary:
                peptide_dictionary[peptide] = 1
                peptide_count += 1
    print("Peptide count: " + str(peptide_count))
    print("Peptide count in large files: " + str(peptide_count_large_files))
    print("Number of files:" + str(len(file_list)))
    print("Number of large files: " + str(num_large_files))
    print("Largest single file: " + str(largest_file))
    print("Name of largest file: " + str(largest_file_name))
    if override_test_file:
        test_file = directory + largest_file_name
    return max_file_number


# Changes a M[16] peptide to X
def change_M16_to_X(peptide):
    peptide = peptide.replace("M[16]", "X")
    return peptide


# Prints peptide duplicates in one file
def print_duplicates(peptides, retention_times):
    for i in range(len(peptides)):
        for j in range(len(peptides)):
            if i != j and peptides[i].sequence == peptides[j].sequence:
                print("Duplicate 1: " + peptides[i].sequence + " " + str(retention_times[i]))
                print("Duplicate 2: " + peptides[j].sequence + " " + str(retention_times[j]))


# Creates array of retention times
def get_retention_times(peptide_data):
    retention_times = []
    for row in range(len(peptide_data)):
        retention_times.append(peptide_data[row].get_retention_time())
    retention_times = np.array(retention_times)
    retention_times = retention_times.reshape(-1, 1)
    return np.array(retention_times)


# Creates 2 dimensional peptide representation based on sequence position and amino acid importance
def create_peptide_representation(peptides, alphabet):
    peptide_representation_array = []
    for i in range(len(peptides)):
        peptide = change_M16_to_X(peptides[i].sequence)
        peptide_representation = np.ones((50, len(alphabet)), dtype=np.float) * 0.0000000001
        for j in range(len(peptide)):
            peptide_representation[j, alphabet.index(peptide[j])] = 1
        peptide_representation_array.append(peptide_representation)
    return np.array(peptide_representation_array)


# Creates extended 2 dimensional peptide representation based on sequence position and amino acid importance
def create_extended_peptide_representation(peptides, alphabet):
    peptide_representation_array = []
    for i in range(len(peptides)):
        peptide = change_M16_to_X(peptides[i].sequence)
        peptide_length = len(peptide)
        peptide_representation = np.ones((100, len(alphabet)), dtype=np.float) * 0.0000000001
        for j in range(len(peptide)):
            peptide_representation[j, alphabet.index(peptide[j])] = 1
            peptide_representation[100 - peptide_length + j, alphabet.index(peptide[j])] = 1
        peptide_representation_array.append(peptide_representation)
    return np.array(peptide_representation_array)


# Generates indexes to compare for the ranked data
def generate_comparison_indexes(number_of_comparisons, number_of_peptides):
    comparison_indexes = []
    for i in range(number_of_comparisons):
        first_peptide = random.randint(0, number_of_peptides - 1)
        second_peptide = random.randint(0, number_of_peptides - 1)
        comparison = [first_peptide, second_peptide]
        comparison_indexes.append(comparison)
    return comparison_indexes


# Creates ranked input data
def generate_ranked_data(peptides, retention_time, comparison_indexes):
    comparison_array = []
    reversed_array = []
    number_of_peptides = len(retention_time)
    for i in range(len(comparison_indexes)):
        first_peptide = comparison_indexes[i][0]
        second_peptide = comparison_indexes[i][1]
        if retention_time[first_peptide] < retention_time[second_peptide]:
            comparison_array.append(np.concatenate((peptides[first_peptide], peptides[second_peptide])))
            reversed_array.append(np.concatenate((peptides[second_peptide], peptides[first_peptide])))
        else:
            comparison_array.append(np.concatenate((peptides[second_peptide], peptides[first_peptide])))
            reversed_array.append(np.concatenate((peptides[first_peptide], peptides[second_peptide])))
    comparison_array.extend(reversed_array)
    return np.array(comparison_array)


# Creates targets for input data
def generate_targets(number_of_comparisons):
    targets = [[1, 0]] * number_of_comparisons + [[0, 1]] * number_of_comparisons
    return np.array(targets)


# Returns list of indexes from the second list for the peptides that are found in both lists
def get_overlapping_peptides(first_peptide_list, second_peptide_list):
    # Check peptide overlap
    first_peptide_sequences = [""] * len(first_peptide_list)
    second_peptide_sequences = [""] * len(second_peptide_list)
    for i in range(len(first_peptide_list)):
        first_peptide_sequences[i] = first_peptide_list[i].sequence
    for i in range(len(second_peptide_list)):
        second_peptide_sequences[i] = second_peptide_list[i].sequence

    all_seq = first_peptide_sequences
    all_seq.extend(second_peptide_sequences)
    unique_seq = set(all_seq)
    # print("Duplicates in first set: " + str(len(first_peptide_sequences) - len(set(first_peptide_sequences))))
    # print("Duplicates in second set: " + str(len(second_peptide_sequences) - len(set(second_peptide_sequences))))
    # print("Shared peptides: " + str(len(all_seq) - len(unique_seq)))

    shared_peptide_indices = []
    for i in range(len(first_peptide_list)):
        for j in range(len(second_peptide_list)):
            if first_peptide_sequences[i] == second_peptide_sequences[j]:
                shared_peptide_indices.append(j)
                break
    return shared_peptide_indices


# Removes specified indices
def remove_indices(the_list, index_list):
    index_list = set(index_list)
    index_list = list(index_list)
    the_list = list(the_list)
    for i in sorted(index_list, reverse=True):
        del the_list[i]
    return np.array(the_list)


# Builds the separate identical network for preprocessing the two inputs
def create_pipeline(input):
    print(input.output_shape)
    network = lasagne.layers.Conv2DLayer(input, num_filters=800, filter_size=(6, len(importance_alphabet)),
                                         nonlinearity=lasagne.nonlinearities.rectify, stride=1)
    # network = lasagne.layers.Conv2DLayer(input, num_filters=64, filter_size = (6, 6), nonlinearity = lasagne.nonlinearities.rectify, stride = 1)
    return network


def build_cnn(input_var=None):
    if extended_peptide_representation:
        l_in = lasagne.layers.InputLayer(shape=(None, 1, 200, len(importance_alphabet)), input_var=input_var)
        first_peptide = create_pipeline(lasagne.layers.SliceLayer(l_in, indices=slice(0, 100), axis=2))
        second_peptide = create_pipeline(lasagne.layers.SliceLayer(l_in, indices=slice(100, 200), axis=2))
    else:
        l_in = lasagne.layers.InputLayer(shape=(None, 1, 100, len(importance_alphabet)), input_var=input_var)
        first_peptide = create_pipeline(lasagne.layers.SliceLayer(l_in, indices=slice(0, 50), axis=2))
        second_peptide = create_pipeline(lasagne.layers.SliceLayer(l_in, indices=slice(50, 100), axis=2))

    network = lasagne.layers.ConcatLayer([first_peptide, second_peptide], axis=1)

    for i in range(10):
        network = lasagne.layers.DenseLayer(network, num_units=64, W=lasagne.init.GlorotUniform('relu'),
                                            nonlinearity=lasagne.nonlinearities.rectify)

    l_out = lasagne.layers.DenseLayer(network, num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
    return l_out


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# Reads test batch peptides and their retention time from file
print("Reading test data")
test_peptides = data_tools.read_data(test_file)
test_retention_times = get_retention_times(test_peptides)

# Counting Peptides
directory = ""
file_list = []
if several_input_files:
    print("Training from multiple files")
    print("Counting unique peptides")
    directory = "/home/daniel/Codes/haydars_code/Data/fdr0001_data/rtime_fdr0.001/"
    max_file_number = count_unique_peptides(directory)
    file_list = get_file_list(directory)

else:
    # Reads training peptides and their retention time from file
    print("Training from one file")
    print("Reading training data")
    peptides = data_tools.read_data(training_file)
    print("Number of training peptides: " + str(len(peptides)))
    print("Removing peptides present in test data")
    shared_peptide_indices = get_overlapping_peptides(test_peptides, peptides)
    peptides = remove_indices(peptides, shared_peptide_indices)
    print("Number of training peptides: " + str(len(peptides)))
    retention_times = get_retention_times(peptides)
    # print_duplicates(peptides, retention_times)

    if extended_peptide_representation:
        peptide_representation = create_extended_peptide_representation(peptides, importance_alphabet)
        print(peptide_representation[0][0])
        print(peptide_representation[0][-1])
    else:
        peptide_representation = create_peptide_representation(peptides, importance_alphabet)

    train_inputs = generate_ranked_data(peptide_representation, retention_times,
                                        generate_comparison_indexes(train_batch_size, len(peptides)))
    train_targets = generate_targets(train_batch_size)

print("Number of test peptides: " + str(len(test_peptides)))
if extended_peptide_representation:
    test_peptide_representation = create_extended_peptide_representation(test_peptides, importance_alphabet)
else:
    test_peptide_representation = create_peptide_representation(test_peptides, importance_alphabet)
test_indexes = generate_comparison_indexes(test_batch_size, len(test_peptides))
test_inputs = generate_ranked_data(test_peptide_representation, test_retention_times, test_indexes)
test_targets = generate_targets(test_batch_size)

# Prepare Theano variables for inputs and targets
input_var = T.tensor4('inputs')
target_var = T.matrix('targets')

# Create neural network model (depending on first command line parameter)
print("Building model and compiling functions...")
network = build_cnn(input_var)

# Create a loss expression for training, i.e., a scalar objective we want
# to minimize (for our multi-class problem, it is the cross-entropy loss):
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
l2_regularization = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
loss = loss + 0.00005 * l2_regularization
loss = loss.mean()
# We could add some weight decay as well here, see lasagne.regularization.

# Create update expressions for training, i.e., how to modify the
# parameters at each training step. Here, we'll use Stochastic Gradient
# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.015, momentum=0.90)

# Create a loss expression for validation/testing. The crucial difference
# here is that we do a deterministic forward pass through the network,
# disabling dropout layers.
test_prediction = lasagne.layers.get_output(network, deterministic=True, allow_input_downcast=True)
test_loss = lasagne.objectives.binary_accuracy(test_prediction, target_var)
test_loss = test_loss.mean()

# Compile a function performing a training step on a mini-batch (by giving
# the updates dictionary) and returning the corresponding training loss:
train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)

# Compile a second function computing the validation loss and accuracy:
# val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)
val_fn = theano.function([input_var, target_var], test_loss, allow_input_downcast=True)

# Finally, launch the training loop.
print("Starting training...")
# We iterate over epochs:
train_errors = [0] * num_epochs
validation_accuracies = [0] * num_epochs
final_accuracy = 0

for epoch in range(num_epochs):
    train_err = 0
    train_batches = 0
    start_time = time.time()

    if several_input_files:
        file_size = 0
        while (file_size < smallest_input_file_size):
            file_number = random.randint(0, max_file_number)
            peptides = data_tools.read_data(directory + file_list[file_number])
            if len(peptides) < smallest_input_file_size:
                continue
            shared_peptide_indices = get_overlapping_peptides(test_peptides, peptides)
            peptides = remove_indices(peptides, shared_peptide_indices)
            file_size = len(peptides)
        retention_times = get_retention_times(peptides)
        if extended_peptide_representation:
            peptide_representation = create_extended_peptide_representation(peptides, importance_alphabet)
        else:
            peptide_representation = create_peptide_representation(peptides, importance_alphabet)

    train_inputs = generate_ranked_data(peptide_representation, retention_times,
                                        generate_comparison_indexes(train_batch_size, len(peptides)))
    train_targets = generate_targets(train_batch_size)

    for batch in iterate_minibatches(train_inputs, train_targets, 1000, shuffle=True):
        inputs, targets = batch
        inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])
        train_err += train_fn(inputs, targets)
        train_batches += 1

    epoch_time = time.time() - start_time

    if epoch % 50 == 0 or num_epochs - epoch <= 20:
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(test_inputs, test_targets, 500, shuffle=False):
            inputs, targets = batch
            inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2])
            err = val_fn(inputs, targets)
            val_acc += err
            val_batches += 1

        train_errors[epoch] = train_err / train_batches
        validation_accuracies[epoch] = val_acc / val_batches

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, epoch_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        if num_epochs - epoch <= 20:
            final_accuracy += val_acc / val_batches * 100

final_accuracy = final_accuracy / 20
print("Final Accuracy: " + str(final_accuracy))
