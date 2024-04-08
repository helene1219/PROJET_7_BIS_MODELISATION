
import pandas as pd
import seaborn as sns
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from bokeh.io import output_notebook, show
from sklearn.metrics import classification_report, confusion_matrix, roc_curve


def data_describe(data):
    data_object={}
    data_object = [(x, data[x].dtype, 
                              data[x].isna().sum().sum(),
                              int(data[x].count())) for x in data.select_dtypes(exclude=['int', 'float'])]
    df_object = pd.DataFrame(data = data_object)
    df_object.columns=['features','dtype','nan','count']
    
    data_numeric = {}
    data_numeric = [(x, data[x].dtype, 
                               int(data[x].isna().sum().sum())/data[x].isnull().count()*100, 
                               int(data[x].count()), 
                               int(data[x].mean()), 
                               round(data[x].std(),1),
                               round(data[x].min(),1), 
                               round(data[x].max(),1)) for x in data.select_dtypes(exclude='object')]    
    df_numeric = pd.DataFrame(data = data_numeric)
    df_numeric.columns=['features','dtype','nan','count', 'mean', 'std', 'min','max']    

    return df_object, df_numeric

def nan_check(data):
    total = data.isnull().sum()
    percent_1 = data.isnull().sum()/data.isnull().count()*100
    percent_2 = (np.round(percent_1, 2))
    missing_data = pd.concat([total, percent_2], 
                             axis=1, keys=['Total', '%']).sort_values('%', ascending=False)
    return missing_data

def plot_stat(data, feature, title) : 
    with sns.color_palette("PiYG"):    
        df=data[data[feature]!="XNA"]
        ax, fig = plt.subplots(figsize=(14,6)) 
        ax = sns.countplot(y=feature, data=df, order=df[feature].value_counts(ascending=False).index)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        for p in ax.patches:
                    percentage = '{:.1f}%'.format(100 * p.get_width()/len(df[feature]))
                    x = p.get_x() + p.get_width()
                    y = p.get_y() + p.get_height()/2
                    ax.annotate(percentage, (x, y), fontsize=14, fontweight='bold')
        ax.spines[['right', 'top']].set_visible(False)
        show(fig)
        
def plot_percent_target1(data, feature) : 
    df=data[data[feature]!="XNA"]
    cat_perc = df[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    with sns.color_palette("PiYG"):  
        ax, fig = plt.subplots(figsize=(14,6)) 
        ax = sns.barplot(y=feature, x='TARGET', data=cat_perc)
        ax.set_title("Répartition défaut de crédit - Target = 1")
        ax.set_xlabel("")
        ax.set_ylabel(" ")

        for p in ax.patches:
                    percentage = '{:.1f}%'.format(100 * p.get_width())
                    x = p.get_x() + p.get_width()
                    y = p.get_y() + p.get_height()/2
                    ax.annotate(percentage, (x, y), fontsize=14, fontweight='bold')
        ax.spines[['right', 'top']].set_visible(False)
        show()
        
#Plot distribution of one feature
def plot_distribution(data,feature, title):
    plt.figure(figsize=(20,6))

    t0 = data.loc[data['TARGET'] == 0]
    t1 = data.loc[data['TARGET'] == 1]

    
    sns.kdeplot(t0[feature].dropna(), color='purple', linewidth=4, label="TARGET = 0")
    sns.kdeplot(t1[feature].dropna(), color='C',  linewidth=4, label="TARGET = 1")
    plt.title(title)
    plt.ylabel('')
    plt.legend()
    show()  

def cf_matrix_roc_auc(model, y_true, y_pred, y_pred_proba, feature_importances):

    fig = plt.figure(figsize=(20,15))
  
    plt.subplot(221)
    cf_matrix = confusion_matrix(y_true, y_pred)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
  
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')

    plt.subplot(222)
    fpr,tpr,_ = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, color='orange', linewidth=5, label='AUC = %0.4f' %roc_auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    if feature_importances : 
        plt.subplot(212)
        indices = np.argsort(model.feature_importances_)[::-1]
    
        features = []
        for i in range(20):
            features.append(X_train.columns[indices[i]])

        sns.barplot(x=features, y=model.feature_importances_[indices[range(20)]], color=("orange"))
        plt.xlabel('Features importance')
        plt.xticks(rotation=90)

        show()
        
def print_score(y_test,y_pred):
    '''Fonction permettant d'afficher les différents scores pertinents pour la classification'''
    print(f'Accuracy score = {accuracy_score(y_test, y_pred)}')
    print(f'Precision score = {precision_score(y_test, y_pred)}')
    print (f'Recall score = {recall_score(y_test,y_pred)}')
    print (f'F1 score = {f1_score(y_test,y_pred)}')
    print (f'ROC AUC score = {roc_auc_score(y_test,y_pred)}')  
    print (f'Score Metier = {Score_metier(y_test,y_pred)}')  