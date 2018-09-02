import tensorflow as tf

# 声明两个变量并计算他们的和
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
resuslt = v1 + v2
init = tf.global_variables_initializer()
server = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    server.save(sess, "model/model.ckpt")

# 生成的文件说明
'''
上述的代码会生成的第一个文件为model.ckpt.meta,它保存了Tensorflow计算图的机构


第二个文件为model.ckpt,这个文件保存了tensorflow程序中每一个变量的取值

最后一个文件为checkpoint文件,这个文件保存了一个目录下所有的模型文件列表.




'''