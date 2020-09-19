import pandas as pd
import numpy as np
from math import ceil
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix,  mean_squared_error, roc_curve, auc
from sklearn.externals.six import StringIO  
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from IPython.display import Image

#used as labels when cutting columns
alph = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
#number of groups for partioning data
education_groups = 4
age_groups = 4
captial_gain_groups = 15
hours_groups = 8

#divide a numerical type into categorical by partitioning the data into n bins
def divideIntoNBins(num_bins, feature, df, max, min):
    bin_size = ceil((max-min)/num_bins)
    bins = []
    for i in range(1,num_bins+1):
        bins.append(i*bin_size)
    return pd.cut(df[feature],bins=num_bins,labels=alph[:num_bins])


if __name__ == "__main__":
    train_df = pd.read_csv("adult_train.csv")
    test_df = pd.read_csv("adult_test.csv")
    #drop unused columns (columns deemed to have low correlation with income)
    train_df = train_df.drop(['Education', 'Native_country','Race', 'Workclass','Fnlwgt','Capital_loss','Marital_status','Occupation'],axis=1)
    test_df = test_df.drop(['Education', 'Native_country','Race', 'Workclass','Fnlwgt','Capital_loss','Marital_status','Occupation'],axis=1)
    
    ###
    ###CLEANING DATA
    ###
    #Remove '.' from the end of adult_test data
    test_df['Yearly_income'] = test_df['Yearly_income'].apply(lambda x: x.split('.')[0])    
    #Replace ' ?' values with NaN
    train_df = train_df.replace(' ?',np.NaN)
    test_df = test_df.replace(' ?',np.NaN)
    nans=0
    #Replace NaN with mean or mode depending on data type
    for col,dtype in train_df.dtypes.items():
        if dtype=='object':
            train_df[col].fillna(train_df[col].mode())
            test_df[col].fillna(test_df[col].mode())
        else:
            train_df[col].fillna(train_df[col].mean())
            test_df[col].fillna(test_df[col].mean())
    ###
    ###PRE-PROCESSING
    ###
    #Divide Education_num into bins
    loss_max = max(train_df['Education_num'].max(), test_df['Education_num'].max())
    loss_min = min(train_df['Education_num'].min(), test_df['Education_num'].min())
    train_df['Education_num'] = divideIntoNBins(education_groups, 'Education_num', train_df, loss_max, loss_min)
    test_df['Education_num'] = divideIntoNBins(education_groups, 'Education_num', test_df, loss_max, loss_min)
    #Divide Age into bins
    age_max = max(train_df['Age'].max(), test_df['Age'].max())
    age_min = min(train_df['Age'].min(), test_df['Age'].min())
    train_df['Age'] = divideIntoNBins(age_groups, 'Age', train_df, age_max, age_min)
    test_df['Age'] = divideIntoNBins(age_groups, 'Age', test_df, age_max, age_min)
    #Divide capital gain into bins
    age_max = max(train_df['Capital_gain'].max(), test_df['Capital_gain'].max())
    age_min = min(train_df['Capital_gain'].min(), test_df['Capital_gain'].min())
    train_df['Capital_gain'] = divideIntoNBins(captial_gain_groups, 'Capital_gain', train_df, age_max, age_min)
    test_df['Capital_gain'] = divideIntoNBins(captial_gain_groups, 'Capital_gain', test_df, age_max, age_min)
    #Hours per week
    age_max = max(train_df['Hours_per_week'].max(), test_df['Hours_per_week'].max())
    age_min = min(train_df['Hours_per_week'].min(), test_df['Hours_per_week'].min())
    train_df['Hours_per_week'] = divideIntoNBins(hours_groups, 'Hours_per_week', train_df, age_max, age_min)
    test_df['Hours_per_week'] = divideIntoNBins(hours_groups, 'Hours_per_week', test_df, age_max, age_min)
    
    #Encode data that can't be converted into a float
    #Encoding using LabelEncoder:
    encoders = {}
    for col,dtype in train_df.dtypes.items():
        if str(dtype)=="object" or str(dtype)=="category":
            #train_df[col] = train_df[col].astype('category')
            #test_df[col] = test_df[col].astype('category')
            #dum_cols.append(col)
            encoders[col] = preprocessing.LabelEncoder()
            encoders[col].fit(train_df[col].values.tolist()+test_df[col].values.tolist())
            train_df[col] = encoders[col].transform(train_df[col].values.tolist())
            test_df[col] = encoders[col].transform(test_df[col].values.tolist())
    #Encoding using dummy variables:
    #dum_cols = []
    #for col,dtype in train_df.dtypes.items():
    #    if str(dtype)=="object" or str(dtype)=="category":
    #        train_df[col] = train_df[col].astype('category')
    #        test_df[col] = test_df[col].astype('category')
    #        dum_cols.append(col)
    #train_df = pd.get_dummies(train_df, columns=dum_cols, drop_first=True)
    #test_df = pd.get_dummies(test_df, columns=dum_cols, drop_first=True)
    #for c in test_df.columns: print(c)

    ###
    ###TRAINING AND TESTING THE MODEL
    ###
    #for label encoder method:
    train_x = train_df.drop('Yearly_income',axis=1).values.tolist()
    train_y = train_df['Yearly_income'].values.tolist()
    test_x = test_df.drop('Yearly_income',axis=1).values.tolist()
    test_y = test_df['Yearly_income'].values.tolist()
    tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=0.4, min_samples_leaf=0.01,max_features=6)

    #for dummy variable method:
    #train_x = train_df.drop('Yearly_income_ >50K',axis=1).values.tolist()
    #train_y = train_df['Yearly_income_ >50K'].values.tolist()
    #test_x = test_df.drop('Yearly_income_ >50K',axis=1).values.tolist()
    #test_y = test_df['Yearly_income_ >50K'].values.tolist()
    #tree_model = DecisionTreeClassifier(criterion='entropy', max_depth=10)

    tree_model.fit(train_x, train_y)
    predictions = tree_model.predict(test_x)
    
    #PRODUCE TREE 
    #for label enocder
    fn=['Relationship','Sex','Age','Education_num','Capital_gain', 'Hours_per_week']
    #for dummy variable
    #fn=train_df.columns.tolist()
    cn=['>50k', '<=50k']
    #fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (13,8), dpi=300)
    #plot_tree(tree_model, max_depth=10, fontsize=10, feature_names = fn, class_names=cn, filled = True)
    #fig.savefig('imagename22.png')


    #PRINT EVALUATION DATA
    corr_matrix = train_df.corr()
    print(corr_matrix["Yearly_income"].apply(lambda x: abs(x)).sort_values(ascending=False))
    print("confusion_matrix:\n", confusion_matrix(test_y, predictions))
    print("precision:\t", precision_score(test_y, predictions, average=None))
    print("recall:\t\t", recall_score(test_y, predictions, average=None))
    print("accuracy:\t", accuracy_score(test_y, predictions))
    print("mse:\t", mean_squared_error(test_y, predictions))