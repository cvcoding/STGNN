import tensorflow as tf
import ptb_reader
from models.dilated_rnn import DilatedRNN
import numpy as np
import math
import time
import pandas as pd

from tensorflow.python import debug as tf_debug

train_data, valid_data, test_data, vocab_size = ptb_reader.ptb_raw_data("./PTB_dataset/")

tf.reset_default_graph()

# Define some hyperparameters
unrolled_dim = 100 
num_of_layers = 5
num_input = 1
number_of_classes = vocab_size # number of different characters
cell_type_list = ["VanillaRNN", "LSTM", "GRU"]
hidden_units = 256 # hidden layer num of features
dilations = [2**j for j in range(num_of_layers)]
batch_size = 128
l_rate = 0.001
number_of_epochs= 4
experiment = "PTB"
decay = 0.9
dropout = 0.1

X_train, y_train = ptb_reader.ptb_producer(train_data, unrolled_dim, vocab_size)
X_val, y_val = ptb_reader.ptb_producer(valid_data, unrolled_dim, vocab_size)
X_test, y_test = ptb_reader.ptb_producer(test_data, unrolled_dim, vocab_size)

X = tf.placeholder(dtype=tf.float32, shape=[None, unrolled_dim, num_input])
y = tf.placeholder(dtype=tf.float32, shape=[None, unrolled_dim, number_of_classes])
batch_size = tf.placeholder(tf.int64)

# Create the dataset
train_dataset = tf.data.Dataset \
        .from_tensor_slices((X, y)) \
        .batch(batch_size, drop_remainder=True)
        # .shuffle(buffer_size=16*batch_size) \

validation_dataset = tf.data.Dataset \
        .from_tensor_slices((X, y)) \
        .batch(batch_size, drop_remainder=True)

test_dataset = tf.data.Dataset \
        .from_tensor_slices((X, y)) \
        .batch(batch_size, drop_remainder=True)

dataset_iterator = tf.data.Iterator.from_structure((tf.float32, tf.float32), ([None, 100, 1], [None, 100, 50]))
batch_X, batch_y = dataset_iterator.get_next()

train_iterator_init = dataset_iterator.make_initializer(train_dataset)
val_iterator_init = dataset_iterator.make_initializer(validation_dataset)
test_iterator_init = dataset_iterator.make_initializer(test_dataset)

for cell_type in cell_type_list:

    print("Starting new optimization process (" + experiment + ")")
    print("Model: Dilated " + cell_type + " for PTB dataset")
    
    # Retrieve the predictions
    pred_object = DilatedRNN(cell_type, hidden_units, dilations)
    output_logits = pred_object.classification(batch_X, number_of_classes, experiment)

    # Loss function
    loss_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_logits, labels=batch_y))/tf.log(tf.constant(2.0))

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
        # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "Dell-XPS-13-9360:6064")
        sess.run(init)

        results_train_set = []
        results_val_set = []

        current_val_loss = math.inf
        
        start = time.time()
        
        for epoch in range(1, number_of_epochs+1):
            
            sess.run(train_iterator_init, feed_dict={ X: X_train, y: y_train, batch_size: 128 })

            count_loss = 0; number_of_batches = 0

            while(True):
                try:
                    # Run optimization
                    batch_loss, _ = sess.run([loss_func, train])
                    
                    count_loss += batch_loss
                    number_of_batches += 1
                    print('Epoch:   {}, Batch:  {}, Loss:   {}'.format(epoch, number_of_batches, batch_loss))
                except tf.errors.OutOfRangeError:
                    break
        
            train_loss = count_loss/number_of_batches
            results_train_set.append((epoch, train_loss))
            
            sess.run(val_iterator_init, feed_dict={ X: X_val, y: y_val, batch_size: len(y_val) })
            val_loss = sess.run(loss_func)
            results_val_set.append((val_loss))
    
            # Check validation loss every 5 epochs
            if epoch % 5 == 0 or epoch == 1:
                # Early stopping and checkpointing
                if val_loss > current_val_loss:
                    saver.restore(sess, "PTB_Checkpoints/Dilated_" + cell_type + ".ckpt")
                    break
                else:
                    current_val_loss = val_loss
                    save_path = saver.save(sess, "PTB_Checkpoints/Dilated_" + cell_type + ".ckpt")
                    print("Model saved in path: %s" % save_path)


                print("Epoch " + str(epoch) + ", Training Loss= " + \
                        "{:.4f}".format(train_loss) + ", Validation Loss= " + \
                        "{:.4f}".format(val_loss))
            
        end = time.time()
        
        training_time = end - start
        
        print("Training Finished!")
        
        print("Training time for this model: ", training_time)
        

        
        sess.run(test_iterator_init, feed_dict={ X: X_test, y: y_test, batch_size: len(y_test) })
        test_loss = sess.run(loss_func)
    
        print("Testing BPC Loss=" + "{:.3f}".format(test_loss))

    # Storing our results to a dataframe
    results_train_set = pd.DataFrame(results_train_set)
    results_val_set = pd.DataFrame(results_val_set)
    
    results = pd.concat([results_train_set, results_val_set], axis=1, join='outer', ignore_index=False)
    results.columns = ["Epochs", "Training Loss", "Validation Loss"]
    
    export_csv = results.to_csv(r"./PTB_results/Dilated_" + cell_type + ".csv", index = None, header=True)
