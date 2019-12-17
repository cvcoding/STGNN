###########################################################################################
### 4.2 Pixel-by-pixel MNIST ###

import numpy as np
import tensorflow as tf
from models.dilated_rnn import DilatedRNN
import time
import math
import pandas as pd
from prepare_data import load_data
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # noqa

# Define some hyperparameters
unrolled_dim = 784 # MNIST data input (img shape: 28*28) # timesteps
num_of_layers = 9
num_input = 1
number_of_classes = 10 # MNIST total classes (0-9 digits)
cell_type_list = ["LSTM", "GRU"]
hidden_units = 96  # hidden layer num of features
dilations = [2**j for j in range(num_of_layers)]
batch_size = 200
l_rate = 0.001
number_of_epochs = 100
experiment = "mnist"
decay = 0.9
permutation_list = [True]

tf.reset_default_graph()

# X_train, y_train = np.load(r"./MNIST_data/X_train.npy"), np.load(r"./MNIST_data/y_train.npy")
# X_val, y_val = np.load(r"./MNIST_data/X_val.npy"), np.load(r"./MNIST_data/y_val.npy")
# X_test, y_test = np.load(r"./MNIST_data/X_test.npy"), np.load(r"./MNIST_data/y_test.npy")

data_path = 'pmnist'  # or pmnist
data_list = load_data(data_path, seq_len=200)
X_train, y_train, X_val, y_val, X_test, y_test = data_list

# # Set the placeholders for our data
X = tf.placeholder(dtype=tf.float32, shape=[None, unrolled_dim, num_input])
y = tf.placeholder(dtype=tf.float32, shape=[None, number_of_classes])
batch_size = tf.placeholder(tf.int64)

# Create the dataset
train_dataset = tf.data.Dataset \
            .from_tensor_slices((X, y)) \
            .shuffle(buffer_size=16*batch_size) \
            .batch(batch_size, drop_remainder=True)

validation_dataset = tf.data.Dataset \
            .from_tensor_slices((X, y)) \
            .batch(batch_size, drop_remainder=True)

test_dataset = tf.data.Dataset \
            .from_tensor_slices((X, y)) \
            .batch(batch_size, drop_remainder=True)

dataset_iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
batch_X, batch_y = dataset_iterator.get_next()

train_iterator = dataset_iterator.make_initializer(train_dataset)
val_iterator = dataset_iterator.make_initializer(validation_dataset)
test_iterator = dataset_iterator.make_initializer(test_dataset)

for permutation in permutation_list:
    
    # if permutation:
    #     np.random.seed(100)
    #     permuted_idx = np.random.permutation(784)
    #     X_train = X_train[:, permuted_idx]
    #     X_val = X_val[:, permuted_idx]
    #     X_test = X_test[:, permuted_idx]

    # X_train = X_train.reshape((-1, unrolled_dim, num_input))
    # X_val = X_val.reshape((-1, unrolled_dim, num_input))
    # X_test = X_test.reshape((-1, unrolled_dim, num_input))
    
    for cell_type in cell_type_list:
        
        if permutation:
            print("Starting new optimization process")
            print("Model: Dilated " + cell_type + " for permuted mnist")
        else:
            print("Starting new optimization process")
            print("Model: Dilated " + cell_type + " for unpermuted mnist")
            
        # Retrieve the predictions
        pred_object = DilatedRNN(cell_type, hidden_units, dilations)
        output_logits = pred_object.classification(batch_X, number_of_classes, experiment)

        # Loss function
        loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_logits, labels=batch_y))

        # Optimizer
        optimizer = tf.train.RMSPropOptimizer(l_rate, decay)
        train = optimizer.minimize(loss_func)

        # number of trainable params
        t_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print('Number of trainable params=', t_params)

        # Compute accuracy of the model
        probabilities = tf.nn.softmax(output_logits)
        predicted_class = tf.argmax(probabilities, 1)
        true_class = tf.argmax(batch_y, 1)
        equality = tf.equal(predicted_class, true_class)
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        with tf.Session() as sess:
    
            sess.run(init)
    
            results_train_set = []
            results_val_set = []
    
            current_val_loss = math.inf
    
            start = time.time()
    
            for epoch in range(1, number_of_epochs+1):

                sess.run(train_iterator, feed_dict={ X: X_train, y: y_train, batch_size: 128 })
                count_loss = 0; number_of_batches = 0; count_accuracy = 0
        
                while(True):
                    try:
                        # Run optimization
                        batch_loss, batch_accuracy, _ = sess.run([loss_func, accuracy, train])
                        
                        count_loss += batch_loss
                        count_accuracy += batch_accuracy
                        number_of_batches += 1
                    except tf.errors.OutOfRangeError:
                        break
            
                train_loss = count_loss/number_of_batches
                train_accuracy = count_accuracy/number_of_batches
                results_train_set.append((epoch, train_loss, train_accuracy))
                
                sess.run(val_iterator, feed_dict={ X: X_val, y: y_val, batch_size: len(y_val) })
                val_loss, val_accuracy = sess.run([loss_func, accuracy])
                results_val_set.append((val_loss, val_accuracy))
        
                # Check validation loss every 5 epochs
                if epoch % 1 == 0 or epoch == 1:

                    print("Epoch " + str(epoch) + ", Training Loss= " + \
                          "{:.4f}".format(train_loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(train_accuracy) + ", Validation Loss= " + \
                          "{:.4f}".format(val_loss) + ", Validation Accuracy= " + \
                          "{:.3f}".format(val_accuracy))

                    # Early stopping and checkpointing
                    if val_loss > current_val_loss:
                        if permutation:
                            saver.restore(sess, "MNIST_Checkpoints/Dilated_" + cell_type + "_permuted.ckpt")
                        else:
                            saver.restore(sess, "MNIST_checkpoints/Dilated_" + cell_type + "_unpermuted.ckpt")
                        # print("Model restored!")
                        # break
                    else:
                        current_val_loss = val_loss
                        if permutation:
                            save_path = saver.save(sess, "MNIST_Checkpoints/Dilated_" + cell_type + "_permuted.ckpt")
                        else:
                            save_path = saver.save(sess, "MNIST_checkpoints/Dilated_" + cell_type + "_unpermuted.ckpt")
                        # print("Model saved in path: %s" % save_path)


            # print("Training Finished!")
            
            end = time.time()
    
            training_time = end - start

            print("Training time for this model: ", training_time)
    
            sess.run(test_iterator, feed_dict={ X: X_test, y: y_test, batch_size: len(y_test) })
            testing_acc = sess.run(accuracy)
        
            print("Testing Accuracy=" + "{:.3f}".format(testing_acc))

    
        # Store our results
        results_train_set = pd.DataFrame(results_train_set)
        results_val_set = pd.DataFrame(results_val_set)
        
        results = pd.concat([results_train_set, results_val_set], axis=1, join='outer', ignore_index=False)
        results.columns = ["Epochs", "Training Loss", "Training Accuracy", "Validation Loss", "Validation Accuracy"]

        if permutation:
            export_csv = results.to_csv (r"MNIST_results/Dilated_" + cell_type + "_permuted.csv", index = None, header=True)
        else:
            export_csv = results.to_csv (r"MNIST_results/Dilated_" + cell_type + "_unpermuted.csv", index = None, header=True)
            
            
################################################### End of script #######################################################