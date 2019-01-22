import tensorflow as tf
https://www.coursera.org/learn/convolutional-neural-networks/programming/bwbJV/convolutional-model-application/discussions/threads/npnNgsUpEeeaSxJ_5wiJJg

def initialize_parameters(beta=0):
  
    tf.set_random_seed(1)                              # so that your "random" numbers match ours
    
    #Regularization
    if beta!=0:
        regularizer = tf.contrib.layers.l2_regularizer(scale=beta)
    else: regularizer=None
    
    W1 = tf.get_variable(..., regularizer=regularizer)
    W2 = tf.get_variable(..., regularizer=regularizer)

    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters, regularizer
	
	
def forward_propagation(X, parameters, regularizer=None):
   
   .......
   
   Z3 = tf.contrib.layers.fully_connected(..., weights_regularizer=regularizer)
  
   return Z3
   
 def compute_cost(Z3, Y, regularizer=None):
    
    .....f
    
    #Regularize
    if regularizer is not None:
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    else:
        reg_term = 0
    
    cost += reg_term
    
    return cost
	
### SAVING
# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in path: %s" % save_path)

### RESTORE

tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("v1", shape=[3])
v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())