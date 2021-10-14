image_condense = tf.nn.pool(
    input = image_detect,
    window_shape= (2,2),
    pooling_type= 'MAX',
    strides=(2,2),
    padding='SAME',
)

plt.figure(figsize=(8, 6))
plt.subplot(121)
plt.imshow(tf.squeeze(image_detect))
plt.axis('off')
plt.title("Detect (ReLU)")
plt.subplot(122)
plt.imshow(tf.squeeze(image_condense))
plt.axis('off')
plt.title("Condense (MaxPool)")
plt.show();
