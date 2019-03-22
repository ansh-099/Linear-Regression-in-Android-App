import tensorflow as tf
session = tf.Session()

# y = Wx +b

x_train = [1.0,2.0,3.0,4.0]
y_train = [-1.0,-2.0,-3.0,-4.0]

W = tf.Variable(initial_value=[1.0],dtype=tf.float32, name='W')
b = tf.Variable(initial_value=[1.0],dtype=tf.float32,name='b')
x = tf.placeholder(dtype=tf.float32,name='x')

y_input = tf.placeholder(dtype=tf.float32,name='y_input')
# y_output = W*x + b

temp = tf.multiply(x=W,y=x,name='multiply')
y_output = tf.add(x=temp,y=b,name='y_output')

loss = tf.reduce_sum(input_tensor=tf.square(x=y_output - y_input),name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01,name='optimizer')
train_step = optimizer.minimize(loss=loss,name='train_step')

# Saver
saver = tf.train.Saver()


session.run(tf.global_variables_initializer())
print(session.run(fetches=loss,feed_dict={x:x_train,y_input:y_train}))

tf.train.write_graph(graph_or_graph_def=session.graph_def,logdir='.',name='linear_regression.pbtxt',as_text=False)

for _ in range(1000):
    session.run(fetches=train_step,feed_dict={x:x_train,y_input:y_train})

saver.save(sess=session,save_path='linear_regression.ckpt')

print(session.run(fetches=[loss,W,b],feed_dict={x:x_train,y_input:y_train}))
print(session.run(fetches=y_output,feed_dict={x:x_train}))



