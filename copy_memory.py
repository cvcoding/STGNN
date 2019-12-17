########################################################################################
    
### 4.1 COPY MEMORT PROBLEM ###

import numpy as np
import tensorflow as tf
from models.dilated_rnn import DilatedRNN
import time
import math
import pandas as pd

class Copy_Memory_Dataset:
    
    @staticmethod
    def generate(num_examples, T):
        data = np.random.randint(low = 0, high = 7, size = (num_examples, 10, 1))
        eights = np.ones((num_examples, T+10, 1), dtype=int) * 8
        nines = np.ones((num_examples, 11, 1), dtype=int) * 9
        X = np.concatenate([data, eights[:, :-11], nines], axis=1)
        Y = np.concatenate([eights, data], axis=1)     
        assert(X.shape == (num_examples, T+20, 1))
        assert(Y.shape == (num_examples, T+20, 1))
        assert((X[:, 10:-11] == Y[:, 10:-11]).all())
        assert((X[:, :10] == Y[:, -10:]).all())
        Y = np.squeeze(np.eye(10)[Y])
        return X, Y

###############################################################################
###############################################################################
        
# Define some hyperparameters
T = [500, 1000]
num_examples = 20000 
num_of_layers = 9
num_input = 1
number_of_classes = 10 # 0-9 digits
cell_type_list = ["VanillaRNN", "LSTM", "GRU"]
hidden_units = 10 # hidden layer num of features
dilations = [2**j for j in range(num_of_layers)]
l_rate = 0.001
number_of_epochs = 20
experiment = "copy_memory"
n_test = 4000
decay = 0.9

tf.reset_default_graph()

for t_len in T:
    
    X_train, y_train = Copy_Memory_Dataset.generate(int(0.7*num_examples), t_len)
    X_val, y_val = Copy_Memory_Dataset.generate(int(0.1*num_examples), t_len)
    X_test, y_test = Copy_Memory_Dataset.generate(int(0.3*num_examples), t_len)

    # # Set the placeholders for our data
    X = tf.placeholder(dtype=tf.float32, shape=[None, X_train.shape[1], num_input])
    y = tf.placeholder(dtype=tf.float32, shape=[None, y_train.shape[1], number_of_classes])
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

    unrolled_dim = t_len + 20
    
    for cell_type in cell_type_list:
        
        print("Starting new optimization process (" + experiment + ")")
        print("Model: Dilated " + cell_type + " for sequence length T= " + '{:d}'.format(t_len))
            
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
        print('Number of trainable params= ' + '{:d}'.format(t_params))

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
                count_loss = 0; number_of_batches = 0
        
                while(True):
                    try:
                        # Run optimization
                        batch_loss, _ = sess.run([loss_func, train])
                        
                        count_loss += batch_loss
                        number_of_batches += 1
                    except tf.errors.OutOfRangeError:
                        break
            
                train_loss = count_loss/number_of_batches
                results_train_set.append((epoch, train_loss))
                
                sess.run(val_iterator, feed_dict={ X: X_val, y: y_val, batch_size: len(y_val) })
                val_loss = sess.run(loss_func)
                results_val_set.append((val_loss))
        
                # Check validation loss every 5 epochs
                if epoch % 5 == 0 or epoch == 1:
                    # Early stopping and checkpointing
                    if val_loss > current_val_loss:
                        saver.restore(sess, "Copy_Memory_Checkpoints/Dilated_" + cell_type + "_" + str(t_len) + ".ckpt")
                        break
                    else:
                        current_val_loss = val_loss
                        save_path = saver.save(sess, "Copy_Memory_Checkpoints/Dilated_" + cell_type + "_" + str(t_len) + ".ckpt")
                        print("Model saved in path: %s" % save_path)


                    print("Epoch " + str(epoch) + ", Training Loss= " + \
                          "{:.4f}".format(train_loss) + ", Validation Loss= " + \
                          "{:.4f}".format(val_loss))
                
            end = time.time()
            
            training_time = end - start
            
            print("Training Finished!")
            
            print("Training time for this model: ", training_time)
                
            sess.run(test_iterator, feed_dict={ X: X_test, y: y_test, batch_size: len(y_test) })
            test_loss = sess.run(loss_func)
        
            print("Testing Cross-Entropy Loss=" + "{:.3f}".format(test_loss))
    

        # Storing our results to a dataframe
        results_train_set = pd.DataFrame(results_train_set)
        results_val_set = pd.DataFrame(results_val_set)
        
        results = pd.concat([results_train_set, results_val_set], axis=1, join='outer', ignore_index=False)
        results.columns = ["Epochs", "Training Loss", "Validation Loss"]
        
        export_csv = results.to_csv(r"./Copy_Memory_results/Dilated_" + cell_type + "_" + str(t_len) + ".csv", index = None, header=True)
        
################################################### End of script #######################################################