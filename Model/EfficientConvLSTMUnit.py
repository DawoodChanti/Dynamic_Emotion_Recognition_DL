
# coding: utf-8

# ### Convolutional LSTM implementation with peephole connections
__author__='Dawood Al Chanti'

# In[1]:

import tensorflow as tf
from tensorflow.contrib.slim import add_arg_scope
from tensorflow.contrib.slim import layers
from tensorflow.contrib.layers.python.layers import initializers

# In[ ]:
 #tf.zeros_initializer: initilize with normal distribution instead of zero states
def init_state(inputs,
               state_shape,
               state_initializer= tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
               dtype=tf.float32):
  """Helper function to create an initial state given inputs.
  Args:
    inputs: input Tensor, at least 2D, the first dimension being batch_size
    state_shape: the shape of the state.
    state_initializer: Initializer(shape, dtype) for state Tensor.
    dtype: Optional dtype, needed when inputs is None.
  Returns:
     A tensors representing the initial state.
  """
  if inputs is not None:
    # Handle both the dynamic shape as well as the inferred shape.
    inferred_batch_size = inputs.get_shape().with_rank_at_least(1)[0]
    batch_size = tf.shape(inputs)[0]
    dtype = inputs.dtype
  else:
    inferred_batch_size = 0
    batch_size = 0

  initial_state = state_initializer(tf.pack([batch_size] + state_shape),dtype=dtype)
  initial_state.set_shape([inferred_batch_size] + state_shape)
  return initial_state



def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    Bias = tf.Variable(initial)
    return Bias

# In[ ]:

@add_arg_scope
def conv_lstm_cell(inputs,
                         state,
                         num_channels,
                         filter_size=3,
                         forget_bias=3.0,
                         scope=None,
                         reuse=None):
    
    spatial_size = inputs.get_shape()[1:3]
    if state is None:
        state = init_state(inputs, list(spatial_size) + [2 * num_channels])
        
    with tf.variable_scope(scope,
                         'BasicConvLstmCell',
                         [inputs, state],
                         reuse=reuse):
        
        
        inputs.get_shape().assert_has_rank(4)
        state.get_shape().assert_has_rank(4)
        c, h = tf.split(3, 2, state)
        
        
        # Learn the biases
        z_i_f_o_bias = bias_variable([num_channels])
        
        z_bias=bias_variable([num_channels])
        i_bias=bias_variable([num_channels])
        f_bias=bias_variable([num_channels])
        o_bias = bias_variable([num_channels])
        
        #inputs_h = tf.concat(3, [inputs, h])
        #--------------------------------------------------------------#
        z_i_f_o_Input = layers.conv2d(inputs=inputs,
                                num_outputs=4 * num_channels, 
                                kernel_size=[filter_size, filter_size],
                                stride=1,
                                activation_fn=None,
                                weights_initializer=initializers.xavier_initializer_conv2d(uniform=True,dtype=tf.float32),
                                scope='Input_Block')

        z_input,i_input,f_input,o_input = tf.split(3, 4, z_i_f_o_Input)
        
        #--------------------------------------------------------------#
        z_i_f_o_Hidden = layers.conv2d(inputs=h,
                                num_outputs=4 * num_channels, 
                                kernel_size=[filter_size, filter_size],
                                stride=1,
                                activation_fn=None,
                                weights_initializer=initializers.xavier_initializer_conv2d(uniform=True,dtype=tf.float32),
                                scope='Gates_Hidden')       

        
        z_h,i_h,f_h,o_h = tf.split(3, 4, z_i_f_o_Hidden)
        
        
        #--------------------------------------------------------------#
        
        i_f_o_C = layers.fully_connected(c, num_outputs= 3 * num_channels,
                                     weights_initializer=initializers.xavier_initializer_conv2d(uniform=True,dtype=tf.float32),
                                     scope='gates_Previous_state')

       
        i_c, f_c, o_c  = tf.split(3, 3, i_f_o_C)

        #--------------------------------------------------------------#
        
        # Input Block 
        z= tf.nn.tanh(z_input + z_h + z_bias)
      
        # Input Gate
        i = tf.nn.sigmoid(i_input + i_h + i_c + i_bias)
        # Forget Gate Gate
        f = tf.nn.sigmoid(f_input + f_h + f_c + f_bias)
        # New Cell Memory State
        new_c = z*i + c*f
        # Output Gate
        o = tf.nn.sigmoid(o_input + o_h + o_c + o_bias) 
        # New output: or hidden state
        new_h =  tf.nn.tanh(new_c)*o
        
        return new_h , tf.concat(3, [new_c, new_h])
