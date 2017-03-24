import collections
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

batch_size = 20
embedding_size = 2
num_sampled = 15

sentences = ["the quick brown fox jumped over the lazy dog",
            "I love cats and dogs",
            "we all love cats and dogs",
            "cats and dogs are great",
            "sung likes cats",
            "she loves dogs",
            "cats can be very independent",
            "cats are great companions when they want to be",
            "cats are playful",
            "cats are natural hunters",
            "It's raining cats and dogs",
            "dogs and cats love sung"]

words = "".join(sentences).split()
count = collections.Counter(words).most_common()
print count[:5]

rdic = [i[0] for i in count]
dic = {w:i for i, w in enumerate(rdic)}
voc_size = len(dic)

data = [dic[word] for word in words]
print ('sample data', data[:10], [rdic[t] for t in data[:10]])

cbow_pairs = []
for i in range(1, len(data)-1):
    cbow_pairs.append([[data[i-1], data[i+1]], data[i]])
print ('context pairs', cbow_pairs[:10])

skip_gram_pairs = []
for c in cbow_pairs:
    skip_gram_pairs.append([c[1], c[0][0]])
    skip_gram_pairs.append([c[1], c[0][1]])
print ('skip_gram pairs', skip_gram_pairs[:10])

def generate_batch(size):
    assert size < len(skip_gram_pairs)
    x_data=[]
    y_data = []
    r = np.random.choice(range(len(skip_gram_pairs)), size, replace=False)
    for i in r:
        x_data.append(skip_gram_pairs[i][0])  # n dim
        y_data.append([skip_gram_pairs[i][1]])  # n, 1 dim
    return x_data, y_data

print ('Batches(x, y)', generate_batch(3))

# Input data
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
# need to shape [batch_size, 1] for nn.nce_loss
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

with tf.device('/cpu:0'):
    embedding = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
    embd = tf.nn.embedding_lookup(embedding, train_inputs)

nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embd, num_sampled, voc_size))

train_op = tf.train.AdamOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # Initializing all variables
    tf.global_variables_initializer().run()

    for step in range(100):
        batch_inputs, batch_labels = generate_batch(batch_size)
        _, loss_val = sess.run([train_op, loss],
                feed_dict={train_inputs: batch_inputs, train_labels: batch_labels})
        if step % 10 == 0:
          print("Loss at ", step, loss_val) # Report the loss

    # Final embeddings are ready for you to use. Need to normalize for practical use
    trained_embeddings = embedding.eval()

if trained_embeddings.shape[1] == 2:
    labels = rdic[:10] # Show top 10 words
    for i, label in enumerate(labels):
        x, y = trained_embeddings[i,:]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
            textcoords='offset points', ha='right', va='bottom')
    plt.savefig("word2vec.png")