import pandas as pd
import numpy as np
import os
from sklearn.utils import class_weight
from keras import regularizers
# from keras.models import Sequential
# from keras.layers import Dense
from itertools import product
import keras.backend as K
from functools import partial
#from keras import utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.layers import Dense
import sys
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, classification_report, average_precision_score
import matplotlib.pyplot as plt
import PIL
from PIL import Image

def create_output_dir(output_path):
    os.makedirs(output_path, exist_ok=True)
    return output_path

def write_stats(filePath, statistics):
    with open(filePath, "w") as text_file:
        text_file.write(statistics)
    print("===== output file : " + filePath)

def print_confusion_matrix(figure_path, y_test, y_pred):
    y_test_class, y_pred_class = y_test, y_pred
    target_names = ["Fast", "Slow"]
    labels_names = [0,1]
    stats = []
    stats.append(classification_report(y_test_class, y_pred_class,labels=labels_names, target_names=target_names, zero_division=0))

    fig, ax = plt.subplots(figsize=(4, 3))

    cm = confusion_matrix(y_test_class, y_pred_class)

    # Calculate ROC-AUC and FPR/FNR
    cm_values = [0 for i in range(4)]
    i = 0
    for row in cm:
        for val in row:
            cm_values[i] = val
            i += 1
    TN, FP, FN, TP = cm_values[0], cm_values[1], cm_values[2], cm_values[3]
    FPR, FNR = round(FP/(FP+TN + 0.1),3), round(FN/(TP+FN  + 0.1),3)
    try:
        ROC_AUC = round(roc_auc_score(y_test, y_pred),3)
    except ValueError:
        ROC_AUC = 0 # if all value are classified into one class, which is BAD dataset
    try:
        PR_AUC = round(average_precision_score(y_test, y_pred),3)
    except ValueError:
        PR_AUC = 0
        
    stats.append("FPR = "+ str(FPR) + "  (" + str(round(FPR*100,1))+ "%)")
    stats.append("FNR = "+ str(FNR) + "  (" + str(round(FNR*100,1))+ "%)")
    stats.append("ROC-AUC = "+ str(ROC_AUC) + "  (" + str(round(ROC_AUC*100,1))+ "%)")
    stats.append("PR-AUC = "+ str(PR_AUC) + "  (" + str(round(PR_AUC*100,1))+ "%)")
    
    disp = ConfusionMatrixDisplay(np.reshape(cm_values, (-1, 2)), display_labels=target_names)
    disp = disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='g')
    ax.set_title("FPR = " + str(round(FPR*100,1))+ "%  and FNR = " + str(round(FNR*100,1))+ "%"); 

    # FN -> bottom left corner
    plt.savefig(figure_path, bbox_inches='tight')
    # print("===== output figure : " + figure_path)
    return stats, ROC_AUC, PR_AUC

def plot_latency_cdf(figure_path, complete_df, title):
    # the df is already readonly IOs
    y_pred = complete_df.loc[complete_df["y_pred"] == 0, "latency"].values
    # Draw CDF
    N=len(y_pred)
    data = y_pred
    # sort the data in ascending order
    x_1 = np.sort(data)
    # get the cdf values of y
    y_1 = np.arange(N) / float(N)

    y_test = complete_df["latency"].values
    N=len(y_test)
    data = y_test
    # sort the data in ascending order
    x_2 = np.sort(data)
    # get the cdf values of y
    y_2 = np.arange(N) / float(N)
    percent_slow = int( (N-len(y_pred)) / N * 100)

    # plotting
    plt.figure(figsize=(6,3))
    plt.xlabel('Latency (us)')
    plt.ylabel('CDF')
    plt.title(title + "; Slow = " + str(percent_slow)+ "%")
    p70_lat = np.percentile(x_2, 70)
    plt.xlim(0, max(p70_lat * 3, 1000)) # Hopefully the x axis limit can catch the tail
    plt.ylim(0, 1) 
    plt.plot(x_2, y_2, label = "Raw Latency", color="red")
    plt.plot(x_1, y_1, label = "FlashNet-powered", linestyle='dashdot', color="green")
    plt.legend(loc="lower right")
    plt.savefig(figure_path, bbox_inches='tight')
    # print("===== output figure : " + figure_path)

    arr_accepted_io = map(str, y_pred)
    return arr_accepted_io

#-------------------------Custom Loss--------------------------
def w_categorical_crossentropy(y_true, y_pred, weights):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    weights = weights.astype(float)
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    cross_ent = tf.keras.losses.categorical_hinge(y_true,y_pred) #K.categorical_crossentropy(y_true,y_pred, from_logits=False)
    return cross_ent * final_mask

#-------------------------Print FP TP FN TN--------------------------
def perf_measure(y_actual, y_pred):
    class_id = set(y_actual).union(set(y_pred))
    TP = []
    FP = []
    TN = []
    FN = []

    for index ,_id in enumerate(class_id):
        TP.append(0)
        FP.append(0)
        TN.append(0)
        FN.append(0)
        for i in range(len(y_pred)):
            if y_actual[i] == y_pred[i] == _id:
                TP[index] += 1
            if y_pred[i] == _id and y_actual[i] != y_pred[i]:
                FP[index] += 1
            if y_actual[i] == y_pred[i] != _id:
                TN[index] += 1
            if y_pred[i] != _id and y_actual[i] != y_pred[i]:
                FN[index] += 1
    total_data = len(y_actual)
    print ("total dataset " + str(total_data))
    print ( "  id  TP  FP  TN   FN")
    for x ,_id in enumerate(class_id):
        print ("  " + str(_id) + "\t" + str(TP[x]) + "\t" +  str(FP[x]) + "\t" +  str(TN[x]) + "\t" +  str(FN[x]))
    
    print ("\n_id    %FP        %FN")
    percentFP = []
    percentFN = []
    for x ,_id in enumerate(class_id):
        if ( FN[x]+ TP[x] > 0 and FP[x]+ TN[x] > 0):
            percentFP.append(FP[x]/( FP[x]+ TN[x])*100)
            percentFN.append(FN[x]/( FN[x]+ TP[x])*100)
            print ("  " + str(_id) + "   " + str(float("{:.2f}".format(percentFP[x]))) + " \t\t " + str(float("{:.2f}".format(percentFN[x]))))
  
    # print (  "\nmacro %FP and %FN = " + str(float("{:.2f}".format(np.sum(percentFP)/2))))
    print ("\n")

train_input_path = sys.argv[1]
percentile_threshold = float(sys.argv[2])

custom_loss = 5.0

train_data = pd.read_csv(train_input_path, dtype='float32',sep=',', header=None)
train_data = train_data.sample(frac=1).reset_index(drop=True)
train_data = train_data.values

train_input = train_data[:,:31]
train_output = train_data[:,31]

lat_threshold = np.percentile(train_output, percentile_threshold)
print("lat_threshold: ",lat_threshold)
num_train_entries = int(len(train_output) * 0.80)
print("num train entries: ",num_train_entries)

train_Xtrn = train_input[:num_train_entries,:]
train_Xtst = train_input[num_train_entries:,:]
train_Xtrn = np.array(train_Xtrn)
train_Xtst = np.array(train_Xtst)

#Classification
train_y = []
for num in train_output:
    labels = [0] * 2
    if num < lat_threshold:
        labels[0] = 1
    else:
        labels[1] = 1
    train_y.append(labels)

p_rejection = train_y.count([0, 1])/train_data.shape[0]

#print(y)
train_ytrn = train_y[:num_train_entries]
train_ytst = train_y[num_train_entries:]
train_ytrn = np.array(train_ytrn)
train_ytst = np.array(train_ytst)

print(type(train_ytrn))
print(type(train_Xtrn))

train_output_train = train_output[num_train_entries:]

w_array = np.ones((2,2))
w_array[1, 0] = custom_loss   #Custom Loss Multiplier
#w_array[0, 1] = 1.2

ncce = partial(w_categorical_crossentropy, weights=w_array)
#----------------------------------------------------------------------
stats = []
model = Sequential()
model.add(Dense(256, input_dim=31, activation='relu'))
model.add(Dense(2, activation='linear'))#,kernel_regularizer=regularizers.l2(0.001)))
model.compile(optimizer='adam', loss=ncce, metrics=['accuracy'])
model_summ = []
model.summary(print_fn=lambda x: model_summ.append(x))
stats += ["\n".join(model_summ)]

# Output Directory
dataset_name = str(Path(os.path.basename(train_input_path)).with_suffix(''))
# model_name = str(Path(os.path.basename(__file__)).with_suffix(''))
parent_dir_name = Path(train_input_path).parent
output_dir = os.path.join(parent_dir_name, dataset_name)   
create_output_dir(output_dir)

# for i in range(8):
model.fit(train_Xtrn, train_ytrn, epochs=8, batch_size=128, verbose=0) 
# print('Iteration '+str(i)+'\n')
# print('On test dataset:\n')

# Evaluation
y_pred = np.argmax(model.predict(train_Xtst), axis=1)
y_test = np.argmax(train_ytst, axis=1)

stats.append('%Profile rejection : '+str(p_rejection))
stats.append('%Model rejection   : '+str(y_pred.tolist().count(True)/len(y_pred)))
    
# Print confusion matrix and stats
figure_path1 = os.path.join(output_dir, "conf_matrix.png")
conf_matrix_stats, ROC_AUC, PR_AUC = print_confusion_matrix(figure_path1, y_test, y_pred)
stats += conf_matrix_stats

outfile_path = os.path.join(output_dir, "eval.stats")
write_stats(outfile_path, "\n".join(stats))

# Plot CDF Figures
# Construct the chosen test set in a dataframe
X_test_df = pd.DataFrame()
X_test_df["latency"] = train_output_train
X_test_df["y_test"] = y_test # Real/True decision
X_test_df["y_pred"] = y_pred 
# LinnOS's features are already read only
figure_path2 = os.path.join(output_dir, "linnos_cdf.png")
title = "Read-IO Latency CDF [ROC-AUC = "+ str(ROC_AUC) + " = " + str(round(ROC_AUC*100,1))+ "%] "
arr_accepted_io = plot_latency_cdf(figure_path2, X_test_df, title)

# Write the accepted IO latencies to csv file
outfile_path = os.path.join(output_dir, "fast_latency.csv")
write_stats(outfile_path, "\n".join(arr_accepted_io))

# Combine all figures
list_im = [figure_path2, figure_path1]
images    = [ Image.open(i) for i in list_im ]
widths, heights = zip(*(i.size for i in images))
# https://stackoverflow.com/questions/30227466/
total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))
x_offset = 0
for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]
figure_path_final = os.path.join(output_dir, "eval.png")
new_im.save(figure_path_final)
print("===== output figure : " + figure_path_final)

count = 0
for layer in model.layers: 
    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]
    name = train_input_path +'.weight_' + str(count) + '.csv' #output_dir_path+'/SSD_31col_iteration_'+str(i)+'_weightcustom1_' + str(count) + '.csv'
    name_b = train_input_path + '.bias_' + str(count) + '.csv' #output_dir_path+'/SSD_31col_iteration_'+str(i)+'_biascustom1_' + str(count) + '.csv'
    np.savetxt(name, weights, delimiter=',')
    np.savetxt(name_b, biases, delimiter=',')
    count += 1
        
# Delete figures after we combine them into a single figure
os.remove(figure_path1)
os.remove(figure_path2)