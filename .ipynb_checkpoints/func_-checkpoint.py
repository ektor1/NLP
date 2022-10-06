
# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.activations import relu, softmax
import keras_tuner as kt
from scikeras.wrappers import KerasClassifier # To use keras with sklearn 
tf.__version__

# LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  

# Scikit Learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# Models
import pickle # to save the models
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import (recall_score, precision_score,
            precision_recall_curve, fbeta_score, make_scorer)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight 

# KERAS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.activations import relu, softmax

# gensim
import gensim
import gensim.downloader as api
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.fasttext import FastText

# Load nltk library for tokenization
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import time # To time the models
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# ===================================================================================================
# ===================================================================================================
# TEXT PREPROCESSING

def preprocess(text):
    """Tokenizes raw text. Removes capitalised text, stopwords, 
    and non-alphabetical characters"""
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()] #restricts string to alphabetic characters only
    return doc

def stemming(sentence):
    """Stems words in sentences using nltk package"""
    ps = PorterStemmer()
    doc = [ps.stem(w) for w in sentence]
    return doc

def lemmatization(sentence):
    """Lemmatises words in sentences using nltk package"""
    lem = WordNetLemmatizer()
    doc = [lem.lemmatize(w) for w in sentence]
    return doc
    
def filter_docs(trainsets, testsets, condition):
    """Removes text given the function condition. The text is removed
    if the condition is true"""
    counter=0
    for data in trainsets:
        number_of_rows = len(data)
        for row,text in enumerate(data):
            if condition(text):
                data.drop([row], axis=0, inplace=True)
                testsets[counter].drop([row], axis=0, inplace=True)
        counter+=1
        print("{} rows removed".format(number_of_rows-len(data)))

def has_vector_representation(word2vec_model, text):
    """check if at least one word in the sentence is in the
    word2vec dictionary"""
    return all(word not in word2vec_model for word in text)
    
# ===================================================================================================
# ===================================================================================================
# AVERAGE VECS FUNCTIONS

def average_vecs(sentence, Model, dimensions):
    """Average word vectors"""
    sumvec = np.zeros((dimensions), dtype='float32') # Initialise array for sentence
    numerator = np.zeros((dimensions), dtype='float32')
    nwords = 0 # Num of words in the sentence that are included in the model
    e = 1e-10 # To prevent division by zero
    for word in sentence:
        if word in Model:
            nwords = nwords + 1.
            numerator += Model[word]
    sumvec = numerator / (nwords+e)
    return sumvec

def tfidf_w2v(sentence, Model, dimensions, tfidf_dict):
    """Profuct of tfidf and word2vec. Returns an array for each sentence"""
    vec = np.zeros((dimensions), dtype='float32') # Initialise array for sentence
    numerator = np.zeros((dimensions), dtype='float32')
    denominator = 0.
    e = 1e-10 # To prevent division by zero
    for word in sentence:
        if word in Model and word in tfidf_dict:
            numerator += tfidf_dict[word][0] * Model[word]
            denominator += tfidf_dict[word][0]
    vec = numerator / (denominator+e)
    return vec

def tfidf_w2v2(sentence, Model, dimensions, tfidf_dict):
    """Product of tfidf and word2vec. Returns an array for each sentence.
    If word in Model and tfidf then takes product of tfidf score and w2v vector. 
    If non of the words in the sentence are in tfidf, then takes applies average vec"""
    vec = np.zeros((dimensions), dtype='float32') # Initialise array for sentence
    numerator = np.zeros((dimensions), dtype='float32')
    denominator = 0.
    nwords = 0
    e = 1e-10 # To prevent division by zero
    for word in sentence:
        if word in Model:
            numerator += Model[word]
            nwords += 1
            if word in tfidf_dict:
                numerator -= Model[word]
                numerator += tfidf_dict[word][0] * Model[word]
                denominator += tfidf_dict[word][0]
    if denominator == 0:
        denominator = nwords
    vec = numerator / (denominator+e)
    return vec
    
# ===================================================================================================
# ===================================================================================================

# Metrics and function of the model with Grid Search
METRICS = [
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
    keras.metrics.AUC(name="auc"), # approximating area under the curve
    keras.metrics.AUC(name="prc", curve="PR"), # precision-recall curve
]

def build_sequential(n_layers, n_neurons, dropout_rate, learning_rate, output_bias):
    """Prepare sequential model for training"""
    output_bias = tf.keras.initializers.Constant(output_bias) # bias layer
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(300,)))
    
    for layer in range(n_layers):      
        model.add(keras.layers.Dense(n_neurons, activation="relu")) 
        
    model.add(keras.layers.Dropout(dropout_rate)) # regularization
    model.add(keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)) # output layer 

    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=opt, 
                  metrics=METRICS) 
    return model
    
    
def build_gs_nn(Xtrain, y_train, Xvalid, y_valid, Xtest, y_test, params_grid):
    """Build one neural network for all the combinations of hyperparameters in the params_grid dictionary"""
    counter=0
    # Dataframe for models and their results
    results = pd.DataFrame(index=range(1), columns=["Model Name", "90%", "95%", "99%", 
                                                    "Layers", "Neurons", "Dropout Rate", "Learning Rate",
                                                    "Epochs", "Batch Size", "Time to Fit (ms)"])
    recall_scores = [0.90, 0.95, 0.99]
    # Calculate class weights
    class_weight = compute_class_weight(class_weight = "balanced", classes= np.unique(y_train), y=y_train)
    class_weight = {0: class_weight[0], 1: class_weight[1]}
    # For bias layer
    unique, counts = np.unique(y_train, return_counts=True)
    neg, pos = counts[0], counts[1] 
    output_bias = np.log([pos/neg])
    
    for n_layers in params_grid['layers']:
        for n_neurons in params_grid['neurons']:
            for dropout_rate in params_grid['dropout_rates']:
                for learning_rate in params_grid['learning_rate']:
                    model = build_sequential(n_layers, n_neurons, dropout_rate, learning_rate, output_bias)
                    
                    for n_epochs in params_grid['epochs']:
                        for batch_size in params_grid['batch_sizes']:
#                             if counter==10:
#                                 break
                            history = model.fit(Xtrain, y_train, 
                                                batch_size=batch_size, epochs=n_epochs, 
                                                class_weight=class_weight, 
                                                validation_data=(Xvalid, y_valid), verbose=0)      
    
                            precision_test, recall_test, threshold_test, timer = best_neural_nets(Xvalid, y_valid, model)
                            
                            # Find precision for recall benchmarks
                            for recall in recall_scores:
                                for row, score in enumerate(recall_test):
                                    if round(score, 2) == recall:
                                        results.loc[counter, str(int(recall*100))+"%"] = precision_test[row]
                                        break
                                    
                            results.loc[counter, "Model Name"] = model        
                            results.loc[counter, "Layers"] = n_layers
                            results.loc[counter, "Neurons"] = n_neurons
                            results.loc[counter, "Dropout Rate"] = dropout_rate
                            results.loc[counter, "Learning Rate"] = learning_rate
                            results.loc[counter, "Epochs"] = n_epochs
                            results.loc[counter, "Batch Size"] = batch_size
                            results.loc[counter, "Time to Fit (ms)"] = timer
                            
                            counter+=1 
                            print(counter) # Tracking progress
    return results
                
def best_neural_nets(X_pred, y_labels, model):
    """Returns precision, recall, and probabilities for predictions"""
    start = time.time()
    y_proba = model.predict(X_pred) 
    end = time.time()
    timer = end-start
    y_proba.round(2)
    y_pred = np.zeros(len(y_proba),).astype("float32") # create array to store results
    
    for row, result in enumerate(y_proba):
        if result >= 0.5:
            y_pred[row] = 1
        else:
            y_pred[row] = 0
    # Calculate recall and precision scores
    test_recall = round(recall_score(y_labels, y_pred),4) 
    test_precision = round(precision_score(y_labels, y_pred),4) 
    
    precision_test, recall_test, threshold_test = precision_recall_curve(y_labels, y_proba) 
    
    return precision_test, recall_test, threshold_test, timer
    
def precision_recall(df):
    """Calculates average of precisions and inputs those scores into a new column"""
    df['Average'] = pd.NaT
    for n_row, row in enumerate(df):
        df['Average'].iloc[n_row] = (df['90%'].iloc[n_row]*0.05 + df['95%'].iloc[n_row]*0.05 + 
                                          df['99%'].iloc[n_row]*0.9) / (0.05+0.05+0.9)
    display(df.sort_values(by="Average", ascending=False).head())
    
    return df
    
# ===================================================================================================
# ===================================================================================================
# SKLEARN MODELS

def classifiers_gs(model_list, model_name, grids, X_train, X_test, X_valid, y_train, y_test, y_valid):
    """Fits classifiers and makes predictions. Then plots precision and recall scores"""
    
    # Dataframe for test and validation sets
    results = pd.DataFrame(index=list(range(0, (len(model_list)))), columns=['Model_Name',
                            'Test_Recall','Test_Precision', 'Valid_Recall','Valid_Precision'])
    best_models = {}
    f2_scorer = make_scorer(fbeta_score, beta=2.0, pos_label=1) # make scoring method. Favors recall 
    counter = 0 # Iterate through each model

    for m in model_list:   
        random_grid = grids[counter] # Use correct grid for each model
        
        clf = RandomizedSearchCV(m, random_grid, scoring=f2_scorer, random_state=42)
        clf.fit(X_train,y_train)
        
        # Fit and store the best model
        params = clf.best_params_
        best = m.set_params(**params)
        best = best.fit(X_train, y_train)
        best_models[model_names[counter]] = best
        filename = model_names[counter] + ".sav"
        pickle.dump(best, open(filename, 'wb')) # save best model according to the scoring method
        
        # Predictions for validation set
        y_valid_pred = best.predict(X_valid)
        
        # Precision/Recall scores
        valid_recall = round(recall_score(y_valid, y_valid_pred),4)
        valid_precision = round(precision_score(y_test, y_test_pred),4)
        
        # Storing recall & precision scores
        results.loc[counter, 'Model_Name'] = model_name
        results.loc[counter, 'Valid_Recall'] = valid_recall
        results.loc[counter, 'Valid_Precision'] = valid_precision
        results.loc[counter, 'Test_Recall'] = test_recall
        results.loc[counter, 'Test_Precision'] = test_precision
        
        counter += 1
        
        display(results) 
        print(best_models, timer)
        
    return results, best_models, timer

def classifiers_gs_svm(model, model_name, grids, X_train, X_test, X_valid, y_train, y_test, y_valid):
    """Fits SVM classifier with different parameters. 
    Stores each model and inputs their recall, precision scores, and time to fit the test set in the results df"""
    
    # Dataframe for test and validation sets
    results = pd.DataFrame(index=range(1), columns=['Model_Name',
                            'Test_Recall','Test_Precision', 'Valid_Recall','Valid_Precision', 'C', 'gamma', 'Time to Fit'])
    
    f2_scorer = make_scorer(fbeta_score, beta=2.0,  pos_label=1) # Make scoring method which favours recall 
    counter = 0 # Iterate through each model

    for num,grid in enumerate(grids):   
        
        clf = RandomizedSearchCV(model, grid, n_iter=1, cv=2, verbose=10, scoring=f2_scorer, 
                                 random_state=42)
        clf.fit(X_train, y_train)
        filename = model_name + str(num) + ".sav"
        pickle.dump(clf, open(filename, 'wb')) # saving each model
        
        # Predictions
        y_valid_pred = clf.predict(X_valid)
        start = time.time()
        y_test_pred = clf.predict(X_test)
        stop = time.time()
        timer = stop-start # time take to fit the test set 
        
        # Precision/Recall scores
        valid_recall = round(recall_score(y_valid, y_valid_pred),4)
        valid_precision = round(precision_score(y_test, y_test_pred),4)
        test_recall = round(recall_score(y_test, y_test_pred),4)
        test_precision = round(precision_score(y_test, y_test_pred),4)
        
        # Storing recall & precision scores
        results.loc[counter, 'Model Name'] = model_name
        results.loc[counter, 'Valid Recall'] = valid_recall
        results.loc[counter, 'Valid Precision'] = valid_precision
        results.loc[counter, 'Test Recall'] = test_recall
        results.loc[counter, 'Test Precision'] = test_precision
        results.loc[counter, 'Time to Fit'] = timer
        
        counter += 1
        
        display(results) 
        
    return results
    
def svm_grid_search(Xtrain, Xvalid, y_train, y_valid, model_name, grids):
    """Fits SVM classifier with different parameters by using for loops. 
    Stores each model parameters and inputs their recall, precision scores, and time to fit the test set in the results df"""
    counter=0 
    # Dataframe models and their results
    results = pd.DataFrame(index=range(1), columns=['Model Name', "90%", "95%", "99%",
                                                     'C', 'gamma', 'Time to Fit (ms)'])
    recall_scores =  [0.90, 0.95, 0.99]
    
    
    for num, params in enumerate(grids):
        m = SVC(random_state=42, probability=True, C=params['C'], gamma=params['gamma'])
        model = m.fit(Xtrain, y_train) 
        
        filename = model_name + str(num) + ".sav"
        pickle.dump(model, open(filename, 'wb')) # saving each model
        
        start = time.time()
        y_proba = model.predict_proba(Xvalid) 
        end = time.time()
        timer = end-start
        y_proba = y_proba[:,1].round(2)
        
        precision_test, recall_test, threshold_test = precision_recall_curve(y_valid, y_proba, pos_label='positive') 
        
        for recall in recall_scores:
            for row, score in enumerate(recall_test):
                if round(score, 2) == recall:
                    results.loc[counter, str(int(recall*100))+"%"] = precision_test[row]
                    break
                
        results.loc[counter, "Model Name"] = filename        
        results.loc[counter, "C"] = params["C"]
        results.loc[counter, "gamma"] = params["gamma"]
        results.loc[counter, "Time to Fit (ms)"] = timer
        
        counter+=1 
        display(results)
        
    return results

def classifiers(model_list, model_name, X_train, X_test, X_valid, y_train, y_test, y_valid):
    """Fits classifiers and makes predictions. Then stores precision and recall scores"""
    
    # Dataframe for test and validation sets
    results = pd.DataFrame(index=list(range(0, (len(model_list)))), columns=['Model_Name',
                            'Test_Recall','Test_Precision', 'Valid_Recall','Valid_Precision'])
    counter = 0 # Iterate through each model
    for m in model_list:
        model = m.fit(X_train, y_train) 
        filename = model_names[counter] + ".sav"
        pickle.dump(model, open(filename, 'wb')) # saving each model
        
        # Predictions
        y_valid_pred = model.predict(X_valid)
        start = time.time()
        y_test_pred = model.predict(X_test)
        stop = time.time()
        timer = stop-start # time take to fit the test set 
        
        # Precision/Recall scores
        valid_recall = round(recall_score(y_valid, y_valid_pred),4)
        valid_precision = round(precision_score(y_test, y_test_pred),4)
        test_recall = round(recall_score(y_test, y_test_pred),4)
        test_precision = round(precision_score(y_test, y_test_pred),4)
        
        # Storing recall & precision scores
        results.loc[counter, 'Model_Name'] = model_name
        results.loc[counter, 'Valid_Recall'] = valid_recall
        results.loc[counter, 'Valid_Precision'] = valid_precision
        results.loc[counter, 'Test_Recall'] = test_recall
        results.loc[counter, 'Test_Precision'] = test_precision
        
        counter += 1
        
    return results, timer
    
# ===================================================================================================
# ===================================================================================================
# VISUALISATIONS

def plotpr(X_test, X_valid, y_test, y_valid, model_filename):
    """Predict probabilities and plot precision/recall curve. filename takes a string-type name"""
    
    model = pickle.load(open(model_filename, 'rb'))
    # Probabilities for each class
    y_test_score = model.predict_proba(X_test)
    y_valid_score = model.predict_proba(X_valid)
    print("y_test_score:", "\n", y_test_score)
    pr_test = y_test_score[:,1] # Store probabilities for positive class
    pr_valid = y_valid_score[:,1]
    precision_test, recall_test, threshold_test = precision_recall_curve(y_test, pr_test, pos_label='positive')
    precision_valid, recall_valid, threshold_valid = precision_recall_curve(y_valid, pr_valid, pos_label='positive')
    
    print(color.BOLD + "Test Set Scores" + color.END, "\n", recall_test[:5],"\n\n", precision_test[:5])
    print("\n") # line break
    print(color.BOLD + "Validation Set Scores" + color.END, "\n", recall_valid[:5],"\n\n", precision_valid[:5])
    
    plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace=0.3)

    plt.subplot(2,1,1) # for test set
    plt.plot(recall_test, precision_test)
    plt.title('Test Set', fontweight='bold', fontsize=11)
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.grid()

    plt.subplot(2,1,2) # for validation set
    plt.plot(recall_valid, precision_valid)
    plt.title('Validation Set', fontweight='bold', fontsize=11)
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.gca().set_ylim(0,1.04)
    plt.gca().set_xlim(-0.04,1.04) 
    plt.grid()

    return plt.show()
    
def pr_curve(recall_test, precision_test):
    """Precision-Recall curve for deep learning models"""
    plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace=0.3)
    plt.subplot(2,1,1) # for test set
    plt.plot(recall_test, precision_test)
    plt.title('Test Set', fontweight='bold', fontsize=11)
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.gca().set_ylim(0,1.04)
    plt.gca().set_xlim(-0.04,1.04) 
    plt.grid()
    
    return plt.show()

def plot_loss(history, n):
    """Plots loss for every epoch"""
    # Use a log scale on y-axis to show the wide range of values.
    plt.semilogy(history.epoch, history.history['loss'],
               color=n, label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'],
               color=n, label='Val ' + label,
               linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.grid()

# ===================================================================================================
# ===================================================================================================
# MISCELLANEOUS

def countvec(vec, model_list, X_train, X_test, X_valid, y_train, y_test, y_valid):
    """Applies count vectorizer from scikit learn to train, test, and valid datasets
    Then fits DecisionTreeClassifier and makes predictions"""
    # Dataframe for test and validation sets
    results = pd.DataFrame(index=list(range(0, (len(model_list)))), columns=['Model Name',
                                                                             'Test_Recall','Test_Precision',
                                                                             'Valid_Recall','Valid_Precision'])
    # Transform the text
    X_train = vec.fit_transform(X_train)
    X_test = vec.transform(X_test)
    X_valid = vec.transform(X_valid)
    
    model_names = ['DecisionTreeClassifier','Support Vector Machine']
    
    print(vec)
    counter = 0
    
    for m in model_list:
        clf = m(random_state=42)
        clf.fit(X_train,y_train)
        
        start = time.time()
        y_test_pred = clf.predict(X_test)
        stop = time.time()
        time = stop-start # time take to fit the test set 
        y_valid_pred = clf.predict(X_valid)
        
        test_recall = round(recall_score(y_test, y_test_pred, pos_label="positive"),4)
        test_precision = round(precision_score(y_test, y_test_pred, pos_label="positive"),4)
        valid_recall = round(recall_score(y_valid, y_valid_pred, pos_label="positive"),4)
        valid_precision = round(precision_score(y_test, y_test_pred, pos_label="positive"),4)
        
        # Storing recall & precision scores
        results.loc[counter, 'Model Name'] = model_names[counter]
        results.loc[counter, 'Test_Recall'] = test_recall
        results.loc[counter, 'Test_Precision'] = test_precision
        results.loc[counter, 'Valid_Recall'] = valid_recall
        results.loc[counter, 'Valid_Precision'] = valid_precision
        
        counter+=1
        
    return results, time
        
    
def hashvec(model_list, X_train, X_test, X_valid, y_train, y_test, y_valid, n_features):
    """Applies HashingVectorizer from scikit learn
    Stores arrays for models in X_hash"""
    # Dataframe for test and validation sets
    results = pd.DataFrame(index=list(range(0, (len(model_list)))), columns=['Model Name',
                                                                             'Test_Recall','Test_Precision',
                                                                             'Valid_Recall','Valid_Precision'])
    vec = HashingVectorizer(n_features=n_features)
    
    # transform the text
    
    X_train = vec.transform(X_train)
    X_test = vec.transform(X_test)
    X_valid = vec.transform(X_valid)
    
    model_names = ['DecisionTreeClassifier','Support Vector Machine']
    
    print(vec)
    counter = 0
    
    for m in model_list:
        clf = m(random_state=42)
        clf.fit(X_train,y_train)
        start = time.time()
        y_test_pred = clf.predict(X_test)
        stop = time.time()
        time = stop-start # time take to fit the test set  
        y_valid_pred = clf.predict(X_valid)
        
        test_recall = round(recall_score(y_test, y_test_pred, pos_label="positive"),4)
        test_precision = round(precision_score(y_test, y_test_pred, pos_label="positive"),4)
        
        valid_recall = round(recall_score(y_valid, y_valid_pred, pos_label="positive"),4)
        valid_precision = round(precision_score(y_test, y_test_pred, pos_label="positive"),4)
        
        # Storing recall & precision scores
        results.loc[counter, 'Model Name'] = model_names[counter]
        results.loc[counter, 'Test_Recall'] = test_recall
        results.loc[counter, 'Test_Precision'] = test_precision
        results.loc[counter, 'Valid_Recall'] = valid_recall
        results.loc[counter, 'Valid_Precision'] = valid_precision
        counter+=1
        
    return results, time

# ===================================================================================================
# ===================================================================================================

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
