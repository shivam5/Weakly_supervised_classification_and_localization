import tensorflow as tf
import numpy as np
import pandas as pd

from detector import Detector
from util import load_image
import os
import random
import ipdb
import pickle
from sklearn import metrics


weight_path = '../../data/caffe_layers_value.pickle'
model_path = '../models/'
checkpoint_file = '../models/checkpoint'
with open(checkpoint_file) as f:
    first_line = f.readline()
print(first_line)
first_line = first_line.split()[1]
first_line = first_line.split("-")[1]
start_epoch = int(first_line.split("\"")[0])
print(start_epoch)
print("Pretrained till epoch", start_epoch)
pretrained_model_path = '../models/model-'+str(start_epoch) 
n_epochs = 10000
init_learning_rate = 0.001
weight_decay_rate = 0.0005
momentum = 0.9
batch_size = 60

dataset_path = '../../dataset/images'
data_entry = '../../dataset/Data_Entry_2017.csv'
train_val_filenames = '../../dataset/train_val_list.txt'
test_filenames = '../../dataset/test_list.txt'

xray_path = '../../data/xray'
trainset_path = '../../data/xray/train.pickle'
valset_path = '../../data/xray/val.pickle'
testset_path = '../../data/xray/test.pickle'
label_dict_path = '../../data/xray/label_dict.pickle'

label_dict = {
    # "No Finding" : 0,
    "Atelectasis" : 0,
    "Cardiomegaly" : 1,
    "Effusion" : 2 ,
    "Infiltration" : 3 ,
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


def label_names_to_encoding(label_names):
    x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for label in label_names:
        if (label != "No Finding"):
            x[label_dict[label]] = 1
    return x

def sigmoid_array(x):
    return 1/ (1+np.exp(-x))

def softmax_array(x):
    """Compute softmax values for each sets of scores in x."""
    return (np.exp(x) / np.sum(np.exp(x), axis=1))



if not os.path.exists( trainset_path ):
    if not os.path.exists( xray_path ):
        os.makedirs( xray_path )

    with open(train_val_filenames) as f:
        train_val_imagelist = f.read()
        train_val_imagelist = train_val_imagelist.splitlines()

    with open(test_filenames) as f:
        test_imagelist = f.read()
        test_imagelist = test_imagelist.splitlines()

    data = pd.read_csv(data_entry)
    sample = os.listdir(dataset_path)
    sample = pd.DataFrame({'Image Index': sample})
    sample = pd.merge(sample, data, how='left', on='Image Index')
    sample.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                      'Patient_Age', 'Patient_Gender', 'View_Position',
                      'Original_Image_Width', 'Original_Image_Height',
                      'Original_Image_Pixel_Spacing_X',
                      'Original_Image_Pixel_Spacing_Y', 'Unnamed']

    sample['Finding_Labels'] = sample['Finding_Labels'].apply(lambda x: x.split('|'))

    sample['Encoding'] = 0
    sample['Encoding'] = sample.apply(lambda row: label_names_to_encoding(row['Finding_Labels']), axis=1)

    sample1 = sample.filter(['Image_Index','Encoding'], axis=1)
    label_dict = sample1.set_index('Image_Index').to_dict().values()[0]

    # We have train_val_imagelist, test_imagelist
    random.shuffle(train_val_imagelist)
    train_len = int(0.9*len(train_val_imagelist))
    train_imagelist = train_val_imagelist[0:train_len]
    val_imagelist = train_val_imagelist[train_len+1:len(train_val_imagelist)]

    # Now, we have train_imagelist, val_imagelist, test_imagelist
    # We have label_dict, which has image names, and corresponding label encoding
    trainset = {x : label_dict[x] for x in train_imagelist}
    valset = {x : label_dict[x] for x in val_imagelist}
    testset = {x : label_dict[x] for x in test_imagelist}

    # print(trainset)

    pickle_out = open(trainset_path,"wb")
    pickle.dump(trainset, pickle_out)
    pickle_out.close()

    pickle_out = open(valset_path,"wb")
    pickle.dump(valset, pickle_out)
    pickle_out.close()

    pickle_out = open(testset_path,"wb")
    pickle.dump(testset, pickle_out)
    pickle_out.close()

    pickle_out = open(label_dict_path,"wb")
    pickle.dump(label_dict, pickle_out)
    pickle_out.close()

    # trainset = pd.DataFrame(trainset)
    # valset = pd.DataFrame(valset)
    # testset = pd.DataFrame(testset)
    # label_dict = pd.DataFrame(label_dict)
    #
    # trainset.to_pickle(trainset_path)
    # valset.to_pickle(valset_path)
    # testset.to_pickle(testset_path)
    # label_dict.to_pickle(label_dict_path)

else:

    pickle_in = open(trainset_path,"rb")
    trainset = pickle.load(pickle_in)

    pickle_in = open(testset_path,"rb")
    testset = pickle.load(pickle_in)

    pickle_in = open(valset_path,"rb")
    valset = pickle.load(pickle_in)

    pickle_in = open(label_dict_path,"rb")
    label_dict = pickle.load(pickle_in)

    # trainset = pd.read_pickle( trainset_path )
    # valset = pd.read_pickle( valset_path )
    # testset  = pd.read_pickle( testset_path )
    # label_dict = pd.read_pickle( label_dict_path )



learning_rate = tf.placeholder( tf.float32, [])
# images_tf = tf.placeholder( tf.float32, [None, 24, 24, 3], name="images")
images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
labels_tf = tf.placeholder( tf.float32, [None, 14], name='labels')
#weight = tf.placeholder( tf.float32, [None, 14], name='labels')


detector = Detector(weight_path, 14)
# detector = Detector(weight_path, n_labels)
print("Detector initializes")

p1,p2,p3,p4,conv5, conv6, gap, output = detector.inference(images_tf)
print("This step done")
#n_pos = tf.cast (tf.count_nonzero(labels_tf, dtype=tf.int32), tf.float32)
#n_neg = tf.cast( tf.subtract( tf.size(labels_tf), tf.count_nonzero(labels_tf, dtype=tf.int32)), tf.float32)
#weights_ = tf.scalar_mul( tf.cast(tf.divide(n_neg, n_pos), tf.float32) , labels_tf)
#print(tf.shape(weights_))
#print(tf.shape(tf.nn.sigmoid_cross_entropy_with_logits( logits = output, labels = labels_tf )))
#weights_ = tf.scalar_mul( tf.constant(10.0), labels_tf)
#loss_tf = tf.reduce_mean( tf.multiply (weights_, tf.nn.sigmoid_cross_entropy_with_logits( logits = output, labels = labels_tf )) )
loss_tf = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( logits = output, labels = labels_tf ))
print("Loss defined")


weights_only = filter( lambda x: x.name.endswith('W:0'), tf.trainable_variables() )
weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only])) * weight_decay_rate
loss_tf += weight_decay

sess = tf.InteractiveSession()
saver = tf.train.Saver( max_to_keep=50 )

optimizer = tf.train.MomentumOptimizer( learning_rate, momentum )
grads_and_vars = optimizer.compute_gradients( loss_tf )
grads_and_vars = map(lambda gv: (gv[0], gv[1]) if ('conv6' in gv[1].name or 'GAP' in gv[1].name) else (gv[0]*0.1, gv[1]), grads_and_vars)
#grads_and_vars = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in grads_and_vars]
train_op = optimizer.apply_gradients( grads_and_vars )
tf.initialize_all_variables().run()

if pretrained_model_path:
    print "Pretrained"
    saver.restore(sess, pretrained_model_path)

f_log = open('../results/log.xray256.txt', 'a')

iterations = 0
loss_list = []
for epoch in range(start_epoch+1, n_epochs):

    print("Training")

    x = []
    y = []
    for key, value in trainset.iteritems():
        x.append(key)
        y.append(value)
    c = list(zip(x, y))
    random.shuffle(c)
    x, y = zip(*c)
    trainset = dict(zip(x, y))

    total_imagelist = []
    total_labels = []
    for key, value in trainset.iteritems():
        total_imagelist.append(key)
        total_labels.append(value)

    for start, end in zip(
        range( 0, len(trainset)+batch_size, batch_size),
        range(batch_size, len(trainset)+batch_size, batch_size)):

        current_imagelist = total_imagelist[start:end]
        current_labels = total_labels[start:end]

        current_labels = np.array(current_labels)
        current_image_paths = map(lambda x: os.path.join(dataset_path, x), current_imagelist )
        current_images = np.array(map(lambda x: load_image(x), current_image_paths))

        _, loss_val, output_val = sess.run(
                [train_op, loss_tf, output],
                feed_dict={
                    learning_rate: init_learning_rate,
                    images_tf: current_images,
                    labels_tf: current_labels
                    })

        loss_list.append( loss_val )

        iterations += 1
        if iterations % 5 == 0:
            print "======================================"
            print "Epoch", epoch, "Iteration", iterations
            print "Processed", start, '/', len(trainset)
            f_log.write('Epoch:'+str(epoch)+'\tIteration:'+str(iterations)+'\t'+ '\n')
            f_log.write('Processed:'+str(start)+'/'+str(len(trainset)) )

            # label_predictions = sigmoid_array((output_val))>0.7
            # label_predictions = label_predictions*1
            #
            # tp = [0]*14
            # fp = [0]*14
            # tn = [0]*14
            # fn = [0]*14
            # for i in range(len(label_predictions)):
            #     pred_labels = label_predictions[i]
            #     actual_labels = current_labels[i]
            #     for j in range(14):
            #         if actual_labels[j] == 1:
            #             if pred_labels[j] ==  1:
            #                 tp[j] = tp[j] + 1
            #             else:
            #                 fn[j] = fn[j] + 1
            #         else:
            #             if pred_labels[j] ==  1:
            #                 fp[j] = fp[j] + 1
            #             else:
            #                 tn[j] = tn[j] + 1
            # tpr = [0]*14
            # fpr = [0]*14
            # for j in range(14):
            #     tpr[j] = tp[j]/(tp[j]+fn[j]+0.001)
            #     fpr[j] = fp[j]/(tn[j]+fp[j]+0.001)
            #
            # # acc = (label_predictions == current_labels).sum()
            # # total = sum(len(x) for x in current_labels)
            # # print "Accuracy:", acc, '/', total
            #
            # for j in range(14):
            #     print "True positive rate",i, ":", tpr[j]
            #     print "False positive rate",i, ":" ,fpr[j]

            print "Training Loss:", np.mean(loss_list)
            print "\n"
            f_log.write('Training Loss'+str(np.mean(loss_list)) + '\n')
            loss_list = []


    n_correct = 0
    n_data = 0
    total_imagelist = []
    total_labels = []
    for key, value in valset.iteritems():
        total_imagelist.append(key)
        total_labels.append(value)

    # tp = [0]*14
    # fp = [0]*14
    # tn = [0]*14
    # fn = [0]*14
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    h = len(thresholds)
    w = 14
    tp = [[0 for x in range(w)] for y in range(h)]
    fp = [[0 for x in range(w)] for y in range(h)]
    tn = [[0 for x in range(w)] for y in range(h)]
    fn = [[0 for x in range(w)] for y in range(h)]

    for start, end in zip(
            range(0, len(valset)+batch_size, batch_size),
            range(batch_size, len(valset)+batch_size, batch_size)
            ):

        current_imagelist = total_imagelist[start:end]
        current_labels = total_labels[start:end]

        current_labels = np.array(current_labels)
        current_image_paths = map(lambda x: os.path.join(dataset_path, x), current_imagelist )
        current_images = np.array(map(lambda x: load_image(x), current_image_paths))

        output_vals = sess.run(
                output,
                feed_dict={images_tf:current_images})

        # print("The output is : ")
        # print(sigmoid_array((output_vals)))
        for it in range(len(thresholds)):
            threshold = thresholds[it]

            label_predictions = sigmoid_array((output_vals))>threshold
            label_predictions = label_predictions*1
            # print "The label predictions for threshold", threshold
            # print(label_predictions)

            # acc = (label_predictions == current_labels).sum()
            # total = sum(len(x) for x in current_labels)
            #
            # n_correct += acc
            # n_data += total
            for i in range(len(label_predictions)):
                pred_labels = label_predictions[i]
                actual_labels = current_labels[i]
                for j in range(14):
                    if actual_labels[j] == 1:
                        if pred_labels[j] ==  1:
                            tp[it][j] = tp[it][j] + 1
                        else:
                            fn[it][j] = fn[it][j] + 1
                    else:
                        if pred_labels[j] ==  1:
                            fp[it][j] = fp[it][j] + 1
                        else:
                            tn[it][j] = tn[it][j] + 1

    tpr = [[0 for x in range(w)] for y in range(h)]
    fpr = [[0 for x in range(w)] for y in range(h)]
    # tpr = [0]*14
    # fpr = [0]*14
    for it in range(len(thresholds)):
        for j in range(14):
            tpr[it][j] = tp[it][j]/(tp[it][j]+fn[it][j]+0.001)
            fpr[it][j] = fp[it][j]/(tn[it][j]+fp[it][j]+0.001)

    auc = [0]*w
    for it in range(w):
        auc[it] = metrics.auc([a[it] for a in fpr], [a[it] for a in tpr])

    # acc = (label_predictions == current_labels).sum()
    # total = sum(len(x) for x in current_labels)
    # print "Accuracy:", acc, '/', total

    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print "Validation set AUC"
    print (auc)
    # for j in range(14):
    #     print "True positive rate",j, ":", tpr[j]
    #     print "False positive rate",j, ":" ,fpr[j]
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

    f_log.write('Validation set AUC'+str(auc)+'\n')



    # acc_all = n_correct / float(n_data)
    # f_log.write('Validation set accuracy : epoch:'+str(epoch)+'\tacc:'+str(acc_all) + '\n')
    # print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    # print 'Validation set accuracy : epoch:'+str(epoch)+'\tacc:'+str(acc_all) + '\n'
    # print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"



    n_correct = 0
    n_data = 0
    total_imagelist = []
    total_labels = []
    for key, value in testset.iteritems():
        total_imagelist.append(key)
        total_labels.append(value)

    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    h = len(thresholds)
    w = 14
    tp = [[0 for x in range(w)] for y in range(h)]
    fp = [[0 for x in range(w)] for y in range(h)]
    tn = [[0 for x in range(w)] for y in range(h)]
    fn = [[0 for x in range(w)] for y in range(h)]

    for start, end in zip(
            range(0, len(testset)+batch_size, batch_size),
            range(batch_size, len(testset)+batch_size, batch_size)
            ):

        current_imagelist = total_imagelist[start:end]
        current_labels = total_labels[start:end]

        current_labels = np.array(current_labels)
        current_image_paths = map(lambda x: os.path.join(dataset_path, x), current_imagelist )
        current_images = np.array(map(lambda x: load_image(x), current_image_paths))

        output_vals = sess.run(
                output,
                feed_dict={images_tf:current_images})

        for it in range(len(thresholds)):
            threshold = thresholds[it]
            label_predictions = sigmoid_array((output_vals))>threshold
            label_predictions = label_predictions*1

            # acc = (label_predictions == current_labels).sum()
            # total = sum(len(x) for x in current_labels)
            #
            # n_correct += acc
            # n_data += total
            for i in range(len(label_predictions)):
                pred_labels = label_predictions[i]
                actual_labels = current_labels[i]
                for j in range(14):
                    if actual_labels[j] == 1:
                        if pred_labels[j] ==  1:
                            tp[it][j] = tp[it][j] + 1
                        else:
                            fn[it][j] = fn[it][j] + 1
                    else:
                        if pred_labels[j] ==  1:
                            fp[it][j] = fp[it][j] + 1
                        else:
                            tn[it][j] = tn[it][j] + 1

    tpr = [[0 for x in range(w)] for y in range(h)]
    fpr = [[0 for x in range(w)] for y in range(h)]
    # tpr = [0]*14
    # fpr = [0]*14
    for it in range(len(thresholds)):
        for j in range(14):
            tpr[it][j] = tp[it][j]/(tp[it][j]+fn[it][j]+0.001)
            fpr[it][j] = fp[it][j]/(tn[it][j]+fp[it][j]+0.001)

    auc = [0]*w
    for it in range(w):
        auc[it] = metrics.auc([a[it] for a in fpr], [a[it] for a in tpr])

    # acc = (label_predictions == current_labels).sum()
    # total = sum(len(x) for x in current_labels)
    # print "Accuracy:", acc, '/', total

    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    print "Test set AUC"
    print (auc)
    # for j in range(14):
    #     print "True positive rate",j, ":", tpr[j]
    #     print "False positive rate",j, ":" ,fpr[j]
    print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

    f_log.write('Test set AUC'+str(auc)+'\n')

    #     acc = (label_predictions == current_labels).sum()
    #     total = sum(len(x) for x in current_labels)
    #
    #     n_correct += acc
    #     n_data += total
    #
    # acc_all = n_correct / float(n_data)
    # f_log.write('Test set accuracy : epoch:'+str(epoch)+'\tacc:'+str(acc_all) + '\n')
    # print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    # print 'Test set accuracy : epoch:'+str(epoch)+'\tacc:'+str(acc_all) + '\n'
    # print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

    saver.save( sess, os.path.join( model_path, 'model'), global_step=epoch)

    init_learning_rate *= 0.99
