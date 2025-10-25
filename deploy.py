from flask import Flask,render_template,request
import pickle
import sys
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict",methods=["GET"])
def predict():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle
    from keras.callbacks import ModelCheckpoint

    sns.set_style("whitegrid")
    data = pd.read_csv("creditcard.csv")
    head = str(data.head())
    

    #data analysis#
    data.info()
    pd.set_option("display.float", "{:.2f}".format)
    describe = data.describe()
    

    #checking for missing values#
    print(data.isnull().sum().sum())
    print(data.columns)
    LABELS = ["Normal", "Fraud"]
    count_classes = pd.value_counts(data['Class'], sort = True)
    count_classes.plot(kind = 'bar', rot=0)
    plt.title("Transaction Class Distribution")
    plt.xticks(range(2), LABELS)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    print(data.Class.value_counts())
    fraud = data[data['Class']==1]
    normal = data[data['Class']==0]
    print(f"Shape of Fraudulant transactions: {fraud.shape}")
    fraudtxn = str(f"Shape of Fraudulant transactions: {fraud.shape}")
    print(f"Shape of Non-Fraudulant transactions: {normal.shape}")
    normaltxn = str(f"Shape of Non-Fraudulant transactions: {normal.shape}")
    print(pd.concat([fraud.Amount.describe(), normal.Amount.describe()], axis=1))
    print(pd.concat([fraud.Time.describe(), normal.Time.describe()], axis=1))

    # plot the time feature
    plt.figure(figsize=(14,10))
    plt.subplot(2, 2, 1)
    plt.title('Time Distribution (Seconds)')
    sns.displot(data['Time'], color='blue')

    #plot the amount feature
    plt.subplot(2, 2, 2)
    plt.title('Distribution of Amount')
    sns.displot(data['Amount'],color='blue')

    # data[data.Class == 0].Time.hist(bins=35, color='blue', alpha=0.6)
    plt.figure(figsize=(14, 12))
    plt.subplot(2, 2, 1)
    data[data.Class == 1].Time.hist(bins=35, color='blue', alpha=0.6, label="Fraudulant Transaction")
    plt.legend()
    plt.subplot(2, 2, 2)
    data[data.Class == 0].Time.hist(bins=35, color='blue', alpha=0.6, label="Non Fraudulant Transaction")
    plt.legend()

    # heatmap to find any high correlations
    plt.figure(figsize=(10,10))
    sns.heatmap(data=data.corr(), cmap="seismic")
    plt.show()

    #data preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    scalar = StandardScaler()
    X = data.drop('Class', axis=1)
    y = data.Class
    X_train_v, X_test, y_train_v, y_test = train_test_split(X, y,test_size=0.3, random_state=42)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train_v, y_train_v,test_size=0.2, random_state=42)
    X_train = scalar.fit_transform(X_train)
    X_validate = scalar.transform(X_validate)
    X_test = scalar.transform(X_test)
    w_p = y_train.value_counts()[0] / len(y_train)
    w_n = y_train.value_counts()[1] / len(y_train)
    print(f"Fraudulant transaction weight: {w_n}")
    fraudwg = str(f"Fraudulant transaction weight: {w_n}")
    print(f"Non-Fraudulant transaction weight: {w_p}")
    normalwg = str(f"Non-Fraudulant transaction weight: {w_p}")

    print(f"TRAINING: X_train: {X_train.shape}, y_train: {y_train.shape}\n{'_'*55}")
    trainx = str(f"TRAINING: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"VALIDATION: X_validate: {X_validate.shape}, y_validate: {y_validate.shape}\n{'_'*50}")
    validx = str(f"VALIDATION: X_validate: {X_validate.shape}, y_validate: {y_validate.shape}")
    print(f"TESTING: X_test: {X_test.shape}, y_test: {y_test.shape}")
    testx = str(f"TESTING: X_test: {X_test.shape}, y_test: {y_test.shape}")

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

    def print_score(label, prediction, train=True):
        if train:
            clf_report = str(pd.DataFrame(classification_report(label, prediction, output_dict=True)))
            print("Train Result:\n================================================")
            print(f"Accuracy Score: {accuracy_score(label, prediction) * 100:.2f}%")
            print("_______________________________________________")
            print(f"Classification Report:\n{clf_report}")
            print("_______________________________________________")
            print(f"Confusion Matrix: \n {confusion_matrix(y_train, prediction)}\n")
            
        elif train==False:
            clf_report = str(pd.DataFrame(classification_report(label, prediction, output_dict=True)))
            print("Test Result:\n================================================")        
            print(f"Accuracy Score: {accuracy_score(label, prediction) * 100:.2f}%")
            print("_______________________________________________")
            print(f"Classification Report:\n{clf_report}")
            print("_______________________________________________")
            print(f"Confusion Matrix: \n {confusion_matrix(label, prediction)}\n") 

    #building ANN model#
    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[-1],)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.summary()

    METRICS = [
    #     keras.metrics.Accuracy(name='accuracy'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]

    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=METRICS)

    callbacks = ModelCheckpoint(filepath="model.pkl",verbose=0,save_best_only=True)
    class_weight = {0:w_p, 1:w_n}

    r = model.fit(
        X_train, y_train, 
        validation_data=(X_validate, y_validate),
        batch_size=2048, 
        epochs=300, 
    #     class_weight=class_weight,
        callbacks=callbacks,
    )
    
    score = model.evaluate(X_test, y_test)
    mscore = str(score)
    plt.figure(figsize=(12, 16))
    plt.subplot(4, 2, 1)
    plt.plot(r.history['loss'], label='Loss')
    plt.plot(r.history['val_loss'], label='val_Loss')
    plt.title('Loss Function evolution during training')
    plt.legend()
    plt.subplot(4, 2, 2)
    plt.plot(r.history['fn'], label='fn')
    plt.plot(r.history['val_fn'], label='val_fn')
    plt.title('Accuracy evolution during training')
    plt.legend()
    plt.subplot(4, 2, 3)
    plt.plot(r.history['precision'], label='precision')
    plt.plot(r.history['val_precision'], label='val_precision')
    plt.title('Precision evolution during training')
    plt.legend()
    plt.subplot(4, 2, 4)
    plt.plot(r.history['recall'], label='recall')
    plt.plot(r.history['val_recall'], label='val_recall')
    plt.title('Recall evolution during training')
    plt.legend()

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print_score(y_train, y_train_pred.round(), train=True)
    print_score(y_test, y_test_pred.round(), train=False)

   

    trscore = {
        'Train Result':{
            f'Accuracy score: {accuracy_score(y_train, y_train_pred.round()) * 100:.2f}%'
        }
    }
    tescore = {
        'Test Result':{
            f'Accuracy score: {accuracy_score(y_test, y_test_pred.round()) * 100:.2f}%'
        }
    }

    tesscore = str(tescore)
    trsscore = str(trscore)


    result = {
        'ANNs': {
            'Train': f1_score(y_train, y_train_pred.round()),
            'Test': f1_score(y_test, y_test_pred.round()),
        },
    }

    processed_text= str(result)
    return render_template("index.html",head = head,describe = describe,fraudtxn = fraudtxn,normaltxn =normaltxn,fraudwg = fraudwg,
                           normalwg = normalwg,trainx = trainx,validx = validx,testx = testx,mscore = mscore,processed_text = processed_text,
                           trsscore = trsscore,tesscore = tesscore)




if __name__ == '__main__':
    app.run(debug=True)