import tensorflow as tf
import pandas as pd
import numpy as np

from detector import Detector
from util import load_image
import pickle
from sklearn import metrics

import skimage.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv

import os
import ipdb
import copy

scale_bbox = 224.0/1024.0
localization_path = '../../data/xray/localization.pickle'
lozalization_data_entry = '../../dataset/BBox_List_2017.csv'


weight_path = '../../data/caffe_layers_value.pickle'
model_path = '../models/'
checkpoint_file = '../models/checkpoint'

# with open(checkpoint_file) as f:
#     first_line = f.readline()
# first_line = first_line.split()[1]
# first_line = first_line.split("-")[1]
# start_epoch = int(first_line.split("\"")[0])
# print("Using model ", start_epoch)
# model_path = '../models/model-'+str(start_epoch)

model_path = '../models/model-19'


dataset_path = '../../dataset/images'
data_entry = '../../dataset/Data_Entry_2017.csv'

train_val_filenames = '../../dataset/train_val_list.txt'
test_filenames = '../../dataset/test_list.txt'

xray_path = '../../data/xray'
trainset_path = '../../data/xray/train.pickle'
valset_path = '../../data/xray/val.pickle'
loc_dict_path = '../../data/xray/test.pickle'

label_dict_path = '../../data/xray/label_dict.pickle'
n_labels=14

output_images_path = "../localization_results"

label_dict = {
    # "No Finding" : 0,
    "Atelectasis" : 0,
    "Cardiomegaly" : 1,
    "Effusion" : 2 ,
    "Infiltrate" : 3 ,
    "Mass" : 4,
    "Nodule" : 5,
    "Pneumonia" : 6 ,
    "Pneumothorax" : 7,
    "Consolidation" : 8,
    "Edema" : 9,
    "Emphysema" : 10 ,
    "Fibrosis" : 11 ,
    "Pleural_Thickening" : 12,
    "Hernia" : 13
}

def label_names_to_integer(label_names):
    x = 0
    for label in label_names:
        if (label != "No Finding"):
            x = x + ( 2**(label_dict[label]) )
    return x


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = (xB - xA + 1) * (yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	if (iou<0):
		return (-1*iou)
	return iou


def label_names_to_encoding(label_names):
    x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for label in label_names:
        if (label != "No Finding"):
            x[label_dict[label]] = 1
    return x

def label_name_to_encoding(label):
        x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if (label != "No Finding"):
            x[label_dict[label]] = 1
        return x

def bbox(img):
        if (np.sum(img)==0):
                return -1,-1,-1,-1
        a = np.where(img==1)
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        return bbox

def sigmoid_array(x):
    return 1/ (1+np.exp(-x))

def softmax_array(x):
    """Compute softmax values for each sets of scores in x."""
    return (np.exp(x) / np.sum(np.exp(x), axis=1))


if not os.path.exists( localization_path ):
        print("Reading localization file")
        loc_dict = {}
        with open(lozalization_data_entry) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                        print(row['Image Index'], row['Finding Label'], row['Bbox [x'], row['y'], row['w'], row['h]'])
                        loc_dict[row['Image Index']] = [label_name_to_encoding(row['Finding Label']), float(row['Bbox [x'])*scale_bbox, float(row['y'])*scale_bbox, float(row['w'])*scale_bbox, float(row['h]'])*scale_bbox ]

        pickle_out = open(localization_path,"wb")
        pickle.dump(loc_dict, pickle_out)
        pickle_out.close()


else:
    pickle_in = open(localization_path,"rb")
    loc_dict = pickle.load(pickle_in)



images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
labels_tf = tf.placeholder( tf.float32, [None, 14], name='labels')
conv_input = tf.placeholder( tf.float32, [None, 14, 14, 1024], name="conv_input")

detector = Detector( weight_path, n_labels )
c1,c2,c3,c4,conv5, conv6, gap, output = detector.inference( images_tf )
classmap = detector.get_classmap( conv_input )

sess = tf.InteractiveSession()
saver = tf.train.Saver()

saver.restore( sess, model_path )
f_log = open('../results/logtest_loc.xray256.txt', 'a')



total_imagelist = []
total_values = []
for key, value in loc_dict.iteritems():
    total_imagelist.append(key)
    total_values.append(value)


batch_size = 1

tp = np.zeros((5,14))
fp = np.zeros((5,14))
fn = np.zeros((5,14))
tn = np.zeros((5,14))
acc = np.zeros((5,14))
afp = np.zeros((5,14))
tiou = [0.1, 0.25, 0.5, 0.75, 0.9]


for start, end in zip(
    range( 0, len(loc_dict)+batch_size, batch_size),
    range(batch_size, len(loc_dict)+batch_size, batch_size)):

    current_imagelist = total_imagelist[start:end]
    current_values = total_values[start:end]
    current_values = current_values[0]
    current_labels = current_values[0]
    x = current_values[1]
    y = current_values[2]
    w = current_values[3]
    h = current_values[4]
    flag = 0

    current_labels = np.array(current_labels)
    current_image_paths = map(lambda x: os.path.join(dataset_path, x), current_imagelist )
    current_images = np.array(map(lambda x: load_image(x), current_image_paths))

    conv6_val, output_val = sess.run(
            [conv6, output],
            feed_dict={
                images_tf: current_images
                })

    true_label = np.argmax(current_labels)
    label_prediction = output_val[0]
    predicted_label = np.argmax(label_prediction)

    print("Actual label is")
    print(true_label)

    print("Predicted label is")
    print(predicted_label)

    print("--------------------------------")


    classmap_vals = sess.run(
            classmap,
            feed_dict={
                conv_input: conv6_val
                })


    min_ = np.amin(classmap_vals)
    max_ = np.amax(classmap_vals)
    classmap_vals = map(lambda x: ((x-min_)/(max_-min_)), classmap_vals)


    filename = current_imagelist[0].split(".")[0]
    filename = filename + "_" + str(predicted_label) + ".png"
    # filename = current_imagelist[0]
    output_file_name = os.path.join(output_images_path, filename)

    cmap = classmap_vals[predicted_label]
    classmap_vis = np.array(cmap)

    current_image = current_images[0]

    heatmap = copy.deepcopy(classmap_vis)
    heatmap[heatmap<0.65] = 0
    heatmap[heatmap>=0.65] = 1
    y1, y2, x1, x2 = bbox(heatmap)

    fix, ax = plt.subplots(1)
    ax.imshow( current_image )
    ax.imshow( classmap_vis, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest' )
    true_bbox = patches.Rectangle((x,y),w,h, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(true_bbox)
    if x1 != -1:
        pred_bbox = patches.Rectangle((x1,y1),x2-x1,y2-y1, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(pred_bbox)


        if (true_label == predicted_label) :
            fix.savefig(output_file_name)

        iou = bb_intersection_over_union([x1, y1, x2, y2], [x, y, x+w, y+h])
        print("The iou between the two boxes is ", iou)

        for j in range(len(tiou)):
            curr_tiou = tiou[j]
            if (true_label == predicted_label) and (iou >= curr_tiou):
                tp[j][true_label] += 1
            elif (true_label == predicted_label) and (iou < curr_tiou) :
                fp[j][true_label] += 1
                flag = 1
    for j in range(len(tiou)):
        if (predicted_label != true_label) :
            fn[j][true_label] += 1
            if flag == 0:
                fp[j][predicted_label] += 1



for i in range(len(tiou)):
    for j in range(14):
        tn[i][j] = len(loc_dict) - tp[i][j] - fp[i][j] - fn[i][j]


acc = (tp+tn)/(tp+fp+fn+tn)
afp = fp/(fp+tn)

print("True positive :  ", tp)
print("False positive :  ", fp)
print("True negative :  ", tn)
print("False negative :  ", fn)

print("The accuracy is ", acc)
print("The average false positive is ", afp)
f_log.write('The accuracy is '+str(acc)+'\n')
f_log.write('The average false positive is '+str(afp)+'\n')
