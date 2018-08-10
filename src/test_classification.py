import tensorflow as tf
import pandas as pd
import numpy as np

from detector import Detector
from util import load_image
import pickle
from sklearn import metrics

import skimage.io
import matplotlib.pyplot as plt

import os
import ipdb

dataset_path = '../../dataset/images'
data_entry = '../../dataset/Data_Entry_2017.csv'
train_val_filenames = '../../dataset/train_val_list.txt'
test_filenames = '../../dataset/test_list.txt'

xray_path = '../../data/xray'
trainset_path = '../../data/xray/train.pickle'
valset_path = '../../data/xray/val.pickle'
testset_path = '../../data/xray/test.pickle'
label_dict_path = '../../data/xray/label_dict.pickle'

weight_path = '../../data/caffe_layers_value.pickle'

checkpoint_file = '../models/checkpoint'

# with open(checkpoint_file) as f:
#     first_line = f.readline()
# first_line = first_line.split()[1]
# first_line = first_line.split("-")[1]
# start_epoch = int(first_line.split("\"")[0])
# print("Using model ", start_epoch)
# model_path = '../models/model-'+str(start_epoch)

model_path = '../models/model-16'



f_log = open('../results/logtest.xray256.txt', 'a')

batch_size = 100

def sigmoid_array(x):
    return 1/ (1+np.exp(-x))

def softmax_array(x):
    """Compute softmax values for each sets of scores in x."""
#    print(x)
#    print(np.exp(x))
#    print(np.sum(np.exp(x), axis=1)
#    print( np.exp(x) / np.sum(np.exp(x), axis=1))
    return (np.exp(x) / np.sum(np.exp(x), axis=1))

pickle_in = open(testset_path,"rb")
testset = pickle.load(pickle_in)

pickle_in = open(valset_path,"rb")
valset = pickle.load(pickle_in)


pickle_in = open(label_dict_path,"rb")
label_dict = pickle.load(pickle_in)

n_labels = 14

# images_tf = tf.placeholder( tf.float32, [None, 24, 24, 3], name="images")
images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
labels_tf = tf.placeholder( tf.float32, [None, 14], name='labels')
conv_input = tf.placeholder( tf.float32, [None, 14, 14, 1024], name="conv_input")

detector = Detector( weight_path, n_labels )
c1,c2,c3,c4,conv5, conv6, gap, output = detector.inference( images_tf )
classmap = detector.get_classmap( conv_input )

sess = tf.InteractiveSession()
saver = tf.train.Saver()

saver.restore( sess, model_path )

total_imagelist = []
total_labels = []
for key, value in testset.iteritems():
    total_imagelist.append(key)
    total_labels.append(value)



# n_correct = 0
# n_data = 0
# total_imagelist = []
# total_labels = []
# for key, value in valset.iteritems():
#     total_imagelist.append(key)
#     total_labels.append(value)
#
# # tp = [0]*14
# # fp = [0]*14
# # tn = [0]*14
# # fn = [0]*14
# thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# h = len(thresholds)
# w = 14
# tp = [[0 for x in range(w)] for y in range(h)]
# fp = [[0 for x in range(w)] for y in range(h)]
# tn = [[0 for x in range(w)] for y in range(h)]
# fn = [[0 for x in range(w)] for y in range(h)]
#
# for start, end in zip(
#         range(0, len(valset)+batch_size, batch_size),
#         range(batch_size, len(valset)+batch_size, batch_size)
#         ):
#
#     print(start, len(valset))
#     current_imagelist = total_imagelist[start:end]
#     current_labels = total_labels[start:end]
#
#     current_labels = np.array(current_labels)
#     current_image_paths = map(lambda x: os.path.join(dataset_path, x), current_imagelist )
#     current_images = np.array(map(lambda x: load_image(x), current_image_paths))
#
#     output_vals = sess.run(
#             output,
#             feed_dict={images_tf:current_images})
#
#     # print("The output is : ")
#     # print(sigmoid_array(output_vals))
#     for it in range(len(thresholds)):
#         threshold = thresholds[it]
#
#         label_predictions = sigmoid_array((output_vals))>threshold
#         label_predictions = label_predictions*1
#         # print "The label predictions for threshold", threshold
#         # print(label_predictions)
#
#         # acc = (label_predictions == current_labels).sum()
#         # total = sum(len(x) for x in current_labels)
#         #
#         # n_correct += acc
#         # n_data += total
#         for i in range(len(label_predictions)):
#             pred_labels = label_predictions[i]
#             actual_labels = current_labels[i]
#             for j in range(14):
#                 if actual_labels[j] == 1:
#                     if pred_labels[j] ==  1:
#                         tp[it][j] = tp[it][j] + 1
#                     else:
#                         fn[it][j] = fn[it][j] + 1
#                 else:
#                     if pred_labels[j] ==  1:
#                         fp[it][j] = fp[it][j] + 1
#                     else:
#                         tn[it][j] = tn[it][j] + 1
#
# tpr = [[0 for x in range(w)] for y in range(h)]
# fpr = [[0 for x in range(w)] for y in range(h)]
# # tpr = [0]*14
# # fpr = [0]*14
# for it in range(len(thresholds)):
#     for j in range(14):
#         tpr[it][j] = tp[it][j]/(tp[it][j]+fn[it][j]+0.001)
#         fpr[it][j] = fp[it][j]/(tn[it][j]+fp[it][j]+0.001)
#
# auc = [0]*w
# for it in range(w):
#     auc[it] = metrics.auc([a[it] for a in fpr], [a[it] for a in tpr])
#
# # acc = (label_predictions == current_labels).sum()
# # total = sum(len(x) for x in current_labels)
# # print "Accuracy:", acc, '/', total
#
# print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
# print "Validation set AUC"
# print (auc)
# # for j in range(14):
# #     print "True positive rate",j, ":", tpr[j]
# #     print "False positive rate",j, ":" ,fpr[j]
# print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
#
# f_log.write('Validation set AUC'+str(auc)+'\n')
#
# # acc_all = n_correct / float(n_data)
# # f_log.write('Validation set accuracy : epoch:'+str(epoch)+'\tacc:'+str(acc_all) + '\n')
# # print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
# # print 'Validation set accuracy : epoch:'+str(epoch)+'\tacc:'+str(acc_all) + '\n'
# # print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
#
#
#
# n_correct = 0
# n_data = 0
# total_imagelist = []
# total_labels = []
# for key, value in testset.iteritems():
#     total_imagelist.append(key)
#     total_labels.append(value)
#
# thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# h = len(thresholds)
# w = 14
# tp = [[0 for x in range(w)] for y in range(h)]
# fp = [[0 for x in range(w)] for y in range(h)]
# tn = [[0 for x in range(w)] for y in range(h)]
# fn = [[0 for x in range(w)] for y in range(h)]
#
# for start, end in zip(
#         range(0, len(testset)+batch_size, batch_size),
#         range(batch_size, len(testset)+batch_size, batch_size)
#         ):
#
#     print(start, len(testset))
#     current_imagelist = total_imagelist[start:end]
#     current_labels = total_labels[start:end]
#
#     current_labels = np.array(current_labels)
#     current_image_paths = map(lambda x: os.path.join(dataset_path, x), current_imagelist )
#     current_images = np.array(map(lambda x: load_image(x), current_image_paths))
#
#     output_vals = sess.run(
#             output,
#             feed_dict={images_tf:current_images})
#
#     for it in range(len(thresholds)):
#         threshold = thresholds[it]
#         label_predictions = sigmoid_array((output_vals))>threshold
#         label_predictions = label_predictions*1
#
#         # acc = (label_predictions == current_labels).sum()
#         # total = sum(len(x) for x in current_labels)
#         #
#         # n_correct += acc
#         # n_data += total
#         for i in range(len(label_predictions)):
#             pred_labels = label_predictions[i]
#             actual_labels = current_labels[i]
#             for j in range(14):
#                 if actual_labels[j] == 1:
#                     if pred_labels[j] ==  1:
#                         tp[it][j] = tp[it][j] + 1
#                     else:
#                         fn[it][j] = fn[it][j] + 1
#                 else:
#                     if pred_labels[j] ==  1:
#                         fp[it][j] = fp[it][j] + 1
#                     else:
#                         tn[it][j] = tn[it][j] + 1
#
# tpr = [[0 for x in range(w)] for y in range(h)]
# fpr = [[0 for x in range(w)] for y in range(h)]
# # tpr = [0]*14
# # fpr = [0]*14
# for it in range(len(thresholds)):
#     for j in range(14):
#         tpr[it][j] = tp[it][j]/(tp[it][j]+fn[it][j]+0.001)
#         fpr[it][j] = fp[it][j]/(tn[it][j]+fp[it][j]+0.001)
#
# auc = [0]*w
# for it in range(w):
#     auc[it] = metrics.auc([a[it] for a in fpr], [a[it] for a in tpr])
#
# # acc = (label_predictions == current_labels).sum()
# # total = sum(len(x) for x in current_labels)
# # print "Accuracy:", acc, '/', total
#
# print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
# print "Test set AUC"
# print (auc)
# # for j in range(14):
# #     print "True positive rate",j, ":", tpr[j]
# #     print "False positive rate",j, ":" ,fpr[j]
# print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
# f_log.write('Test set AUC'+str(auc)+'\n')

batch_size = 1

for start, end in zip(
    range( 0, len(testset)+batch_size, batch_size),
    range(batch_size, len(testset)+batch_size, batch_size)):

    current_imagelist = total_imagelist[start:end]
    current_labels = total_labels[start:end]

    current_labels = np.array(current_labels)
    current_image_paths = map(lambda x: os.path.join(dataset_path, x), current_imagelist )
    current_images = np.array(map(lambda x: load_image(x), current_image_paths))

    conv6_val, output_val = sess.run(
            [conv6, output],
            feed_dict={
                images_tf: current_images
                })


    threshold = 0.5
    label_predictions = sigmoid_array((output_val))>threshold
    label_predictions = label_predictions*1

    print("Actual label is")
    print(current_labels)

    print("Predicted label is")
    print(label_predictions)

    print("Scores are")
    print(sigmoid_array((output_val)))

    print("--------------------------------")


#    print("The shape of conv6 is", conv6.shape)
#    print("The shape of image_tf is", images_tf.shape)
#    print("The shape of labels_tf is", labels_tf.shape)
#    print("The shape of conv_input is", conv_input.shape)
#    print("The shape of conv6_val is", conv6_val.shape)

    classmap_vals = sess.run(
            classmap,
            feed_dict={
                conv_input: conv6_val
                })


    # classmap_answer = sess.run(
    #         classmap,
    #         feed_dict={
    #             labels_tf: current_labels,
    #             conv6: conv6_val
    #             })

    # print(classmap_vals)
    # print(classmap_vals.shape)

    true_label = label_predictions[0]
    min_ = np.amin(classmap_vals)
    max_ = np.amax(classmap_vals)
    classmap_vals = map(lambda x: ((x-min_)/(max_-min_)), classmap_vals)
    for i in range(len(true_label)):
        if true_label[i] == 1:
            cmap = classmap_vals[i]
            classmap_vis = np.array(cmap)

            # print(current_images.shape)
            current_image = current_images[0]

            # print (classmap_vis)
            plt.imshow( current_image )
            plt.imshow( classmap_vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest' )
            plt.show()
    # for cmap in classmap_vals:
    #     # classmap = np.stack((classmap,)*3, -1)
    #
    #     # Uncomment this part for visualizing
    #     # classmap_vis = map(lambda x: ((x-x.min())/(x.max()-x.min())), cmap)
    #     # cmap[cmap > 0.75] = 0
    #     # cmap[cmap < 0.25] = 0
    #     classmap_vis = np.array(cmap)
    #
    #     # print(current_images.shape)
    #     current_image = current_images[0]
    #
    #     # print (classmap_vis)
    #     plt.imshow( current_image )
    #     plt.imshow( classmap_vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest' )
    #     plt.show()
