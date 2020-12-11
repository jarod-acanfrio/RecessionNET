import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

NUM_MODEL_RUNS = 1 #abs(int(input("Enter the number of times to train model and store metrics: ")))
NUM_EPOCHS = 100 #abs(int(input("Enter the number of epochs to run: ")))
BATCH_SIZE = 32 #abs(int(input("Enter the desired batch size: ")))

LAYER_1_NEURONS = 512 #abs(int(input("Enter the number of neurons for layer 1: ")))
LAYER_2_NEURONS = 128 #abs(int(input("Enter the number of neurons for layer 2: ")))
LAYER_3_NEURONS = 64 #abs(int(input("Enter the number of neurons for layer 3: ")))

LOOK_BACK = 3 #abs(int(input("Enter the number of previous months to use for predictions: ")))
LOOK_AHEAD = 1 #abs(int(input("Enter the number of months in the future to predict a recession at: ")))

TRAIN_PERCENTAGE = 0.7 #abs(float(input("Enter the percentage of data to use in the training set: ")))
DROPOUT_RATE = 0.1 #abs(float(input("Enter a dropout rate for regularization of the recurrent layers. Value in range [0, 1]: ")))
CLASSIFIER_THRESHOLD = [0.04] #abs(float(input("Enter threshold for classifer: ")))
TO_PRINT = False #bool(input("Would you like to print the Train/Val curve and ROC every iteration? (0=False, 1=True): "))
WRITE_METRICS = False #bool(input("Would you like to write the output metrics to Excel .csv files? (0=False, 1=True): "))
DISPLAY_MODEL = False #bool(input("Would you like to display model summary? (0=False, 1=True): "))

NUM_FEATURES = 10
SCALER = MinMaxScaler()
TO_SCALE = True
OPTIMIZER = tf.keras.optimizers.Adam()
LOSS = tf.keras.losses.BinaryCrossentropy()
DENSE_ACTIVATION = "sigmoid"
#EARLY_STOP = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min') 
METRICS = [
      #tf.keras.metrics.TruePositives(name='tp'),
      #tf.keras.metrics.FalsePositives(name='fp'),
      #tf.keras.metrics.TrueNegatives(name='tn'),
      #tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      #tf.keras.metrics.Precision(name='precision'),
      #tf.keras.metrics.Recall(name='recall'),
      #tf.keras.metrics.AUC(name='auc'),
]

def plot_training(model, name):

    plt.plot(model.history['loss'], label='train loss')
    plt.plot(model.history['accuracy'], label='train accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Value of Metric')
    plt.title('{} RecessionNET Training/Validation Loss and Accuracy vs Epoch'.format(name))
    plt.legend()
    plt.show()

    return(0)

def plot_predictions(actual, predictions,  name):

    plt.plot(actual, label='Actual')
    plt.plot(predictions, label='Predictions')

    plt.xlabel('Time')
    plt.ylabel('Probability of Recession')
    plt.title('{} RecessionNET Actual/Predicted vs Time'.format(name))
    plt.legend()
    plt.show()

    return(0)

def plot_out_of_sample_ROC(actual, predictions, name):
    
    fpr, tpr, thresholds = metrics.roc_curve(actual, predictions)
    auc = auc(fpr, tpr)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Area: {:.3f})'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('{} RecessionNET ROC curve'.format(name))
    plt.legend(loc='best')
    plt.show()

    return(0)

def final_metrics(actual, predictions, threshold):

    binary_predictions = np.where(predictions > threshold, 1, 0)

    accuracy = metrics.accuracy_score(actual, binary_predictions)
    auc = metrics.roc_auc_score(actual, binary_predictions)
    f1 = metrics.f1_score(actual, binary_predictions)
    recall = metrics.recall_score(actual, binary_predictions)
    
    if TO_PRINT:
        print("Out-of-sample Accuracy: {}".format(accuracy))
        print("Out-of-sample Area Under Receiver Operating Curve: {}".format(auc))
        print("Out-of-sample F1-Score: {}".format(f1))
        print("Out-of-sample Recall: {}\n".format(recall))

    return [accuracy, auc, f1, recall]
    

def make_dataset(features, responses, look_back, look_ahead):
    print("\nGenerating dataset...")
    print("Using the past {} months of data to predict a recession {} month(s) away.".format(look_back, look_ahead))

    features = np.array(features)
    responses = np.array(responses)

    # Normalize Sets Here
    if TO_SCALE:
        features = SCALER.fit_transform(features)
        responses = SCALER.fit_transform(responses)

    feature_set, response_set = [], []
    t = 0
    max_index = len(features)-1
    for t in range(max_index-look_back-look_ahead+2):
        feature_sample = features[t:t+look_back, ]
        response_sample = responses[t+look_back+look_ahead-1, ]
        feature_set.append(feature_sample)
        response_set.append(response_sample)

    print("Dataset has been made.\n")

    return np.array(feature_set), np.array(response_set)

def split_dataset(features, responses, split_ratio):
    print("\nSplitting data...\n")
    n = len(features)
    print("Length of data set {}".format(n))
    print("Length of train set {}".format(int(n*split_ratio)))
    print("Length of test set {}".format(int(n-(n*split_ratio)+1)))
    X_train = features[0:int(n * split_ratio)]
    Y_train = responses[0:int(n * split_ratio)]
    X_test = features[int(n * split_ratio)+1:]
    Y_test = responses[int(n * split_ratio)+1:]
    print("Dataset has been split into test-train sets.\n")
    return X_train, Y_train, X_test, Y_test

print("\nImporting data...\n")
macrovars = pd.read_csv("macrovars.csv")
recession = pd.read_csv("recession_state.csv")
macrovars = macrovars.rename(columns={"SP500 (Percent change from a year ago)": "SP5001YDelta", 
                                      "10Y Rate - 3M TBill ": "10YR3MDelta",
                                      "10Y Rate (Percent change from a year ago": "10YR1YDelta",
                                      "3M TBill (Percent Change from a year ago)": "3M1YDelta",
                                      "3M TBill - FF ": "3MFFDelta",
                                      "6M TBill - FF": "6MFFDelta",
                                      "1Y Rate - FF": "1YRFFDelta",
                                      "5Y Rate - FF": "5YRFFDelta",
                                      "10Y Rate - FF": "10YRFFDelta",
                                      "Composite Leading Indicator": "CLI"})

macrovars = macrovars.drop(["Date"], axis = 1)
recession = recession.rename(columns={"Recession State": "Recession"})

print("Features:")
print(macrovars.head())

print("\nResponse:")
print(recession.head())

macrovars = macrovars.to_numpy()
recession = recession.to_numpy()

X, Y = make_dataset(features = macrovars, responses = recession, look_back = LOOK_BACK, look_ahead = LOOK_AHEAD)
X_train, Y_train, X_test, Y_test = split_dataset(features = X, responses = Y, split_ratio = TRAIN_PERCENTAGE)

print("X_train shape: {}".format(X_train.shape))
print("Y_train shape: {}".format(Y_train.shape))
print("\n")

###################################################################################################################################################

for j in range(len(CLASSIFIER_THRESHOLD)):

    LSTM_HISTORY = []
    GRU_HISTORY = []

    for i in range(NUM_MODEL_RUNS):

        inputs = tf.keras.Input(shape=(LOOK_BACK, NUM_FEATURES))
        inputs.shape
        lstm1 = tf.keras.layers.LSTM(units = LAYER_1_NEURONS, input_shape = (LOOK_BACK, NUM_FEATURES), return_sequences = True, dropout=DROPOUT_RATE)
        lstm2 = tf.keras.layers.LSTM(units = LAYER_2_NEURONS, input_shape = (LOOK_BACK, NUM_FEATURES), return_sequences = True, dropout=DROPOUT_RATE)
        lstm3 = tf.keras.layers.LSTM(units = LAYER_3_NEURONS, input_shape = (LOOK_BACK, NUM_FEATURES), return_sequences = False, dropout=DROPOUT_RATE)

        x = lstm1(inputs)
        x = lstm2(x)
        x = lstm3(x)
        outputs = tf.keras.layers.Dense(units = 1, activation=DENSE_ACTIVATION)(x)
        lstm_model = tf.keras.Model(inputs = inputs, outputs = outputs)

        if DISPLAY_MODEL:
            lstm_model.summary()
            tf.keras.utils.plot_model(lstm_model, show_shapes=True, to_file = "lstm_model.png")
        lstm_model.compile(loss=LOSS,
                        optimizer=OPTIMIZER,
                        metrics=METRICS)
         
        lstm_model_history = lstm_model.fit(x = X_train, y = Y_train, epochs = NUM_EPOCHS, verbose = 0) #validation_data = (X_test, Y_test), callbacks = [EARLY_STOP])
        lstm_model_preds = lstm_model.predict(x = X_test)
       
        # plot training history
        if TO_PRINT:
            plot_training(lstm_model_history, "LSTM")
        
        Y_predicted = lstm_model.predict(x = X_test).ravel()
        plot_predictions(Y_test, Y_predicted, "LSTM")
        oos_metrics = final_metrics(Y_test, Y_predicted, CLASSIFIER_THRESHOLD[j])
        LSTM_HISTORY.append(oos_metrics)

        if TO_PRINT:
            plot_out_of_sample_ROC(Y_test, Y_predicted, "LSTM")

        #########################################################################################################################################################################

        inputs = tf.keras.Input(shape=(LOOK_BACK, NUM_FEATURES))
        inputs.shape
        gru1 = tf.keras.layers.GRU(units = LAYER_1_NEURONS, input_shape = (LOOK_BACK, NUM_FEATURES), return_sequences = True, dropout=DROPOUT_RATE)
        gru2 = tf.keras.layers.GRU(units = LAYER_2_NEURONS, input_shape = (LOOK_BACK, NUM_FEATURES), return_sequences = True, dropout=DROPOUT_RATE)
        gru3 = tf.keras.layers.GRU(units = LAYER_3_NEURONS, input_shape = (LOOK_BACK, NUM_FEATURES), return_sequences = False, dropout=DROPOUT_RATE)

        x = gru1(inputs)
        x = gru2(x)
        x = gru3(x)
        outputs = tf.keras.layers.Dense(units = 1, activation=DENSE_ACTIVATION)(x)
        gru_model = tf.keras.Model(inputs = inputs, outputs = outputs)

        if DISPLAY_MODEL:
            gru_model.summary()
            tf.keras.utils.plot_model(gru_model, show_shapes=True, to_file = "gru_model.png")

        gru_model.compile(loss=LOSS,
                        optimizer=OPTIMIZER,
                        metrics=METRICS)

        gru_model_history = gru_model.fit(x = X_train, y = Y_train, epochs = NUM_EPOCHS, verbose = 0) #validation_data = (X_test, Y_test), callbacks = [EARLY_STOP])
        gru_model_preds = gru_model.predict(x = X_test)

        if TO_PRINT:
            plot_training(gru_model_history, "GRU")

        Y_predicted = gru_model.predict(x = X_test).ravel()
        plot_predictions(Y_test, Y_predicted, "GRU")
        oos_metrics = final_metrics(Y_test, Y_predicted, CLASSIFIER_THRESHOLD[j])
        GRU_HISTORY.append(oos_metrics)
        
        if TO_PRINT:
            plot_out_of_sample_ROC(Y_test, Y_predicted, "GRU")

        tf.keras.backend.clear_session()

    if WRITE_METRICS:

        LSTM_HISTORY_df = pd.DataFrame(LSTM_HISTORY, columns = ["Accuracy", "AUC", "F1-Score", "Recall"])
        GRU_HISTORY_df = pd.DataFrame(GRU_HISTORY, columns = ["Accuracy", "AUC", "F1-Score", "Recall"])

        LSTM_HISTORY_df.to_csv("LSTM_HISTORY_{}.csv".format(str(CLASSIFIER_THRESHOLD[j])))
        GRU_HISTORY_df.to_csv("GRU_HISTORY_{}.csv".format(str(CLASSIFIER_THRESHOLD[j])))

print("RecessionNET has completed running.")
print("All output metrics have been written into the current working directory.")
