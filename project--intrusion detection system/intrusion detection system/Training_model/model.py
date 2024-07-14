# Multinomial Classification (normal or DOS or PROBE or R2L or U2R)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
import pickle
import statsmodels.formula.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from sklearn import datasets
from sklearn.feature_selection import RFE
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn import metrics
import sklearn.tree as dt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

train=pd.read_csv('NSL_Dataset\Train.txt',sep=',')
test=pd.read_csv('NSL_Dataset\Test.txt',sep=',')

# print(train.head())
# print(test.head())

columns=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
         "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
         "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate",
         "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
         "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
         "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
         "dst_host_srv_rerror_rate","attack","last_flag"]
train.columns=columns
test.columns=columns
# print(train.head())
# print(test.head())
# print(train.info())
# print(test.info())
# print(train.describe().T)

# In attack_class normal means 0, DOS means 1, PROBE means 2, R2L means 3 and U2R means 4.
train.loc[train.attack=='normal','attack_class']=0

train.loc[(train.attack=='back') | (train.attack=='land') | (train.attack=='pod') | (train.attack=='neptune') |
         (train.attack=='smurf') | (train.attack=='teardrop') | (train.attack=='apache2') | (train.attack=='udpstorm') |
         (train.attack=='processtable') | (train.attack=='worm') | (train.attack=='mailbomb'),'attack_class']=1

train.loc[(train.attack=='satan') | (train.attack=='ipsweep') | (train.attack=='nmap') | (train.attack=='portsweep') |
          (train.attack=='mscan') | (train.attack=='saint'),'attack_class']=2

train.loc[(train.attack=='guess_passwd') | (train.attack=='ftp_write') | (train.attack=='imap') | (train.attack=='phf') |
          (train.attack=='multihop') | (train.attack=='warezmaster') | (train.attack=='warezclient') | (train.attack=='spy') |
          (train.attack=='xlock') | (train.attack=='xsnoop') | (train.attack=='snmpguess') | (train.attack=='snmpgetattack') |
          (train.attack=='httptunnel') | (train.attack=='sendmail') | (train.attack=='named'),'attack_class']=3

train.loc[(train.attack=='buffer_overflow') | (train.attack=='loadmodule') | (train.attack=='rootkit') | (train.attack=='perl') |
          (train.attack=='sqlattack') | (train.attack=='xterm') | (train.attack=='ps'),'attack_class']=4

test.loc[test.attack=='normal','attack_class']=0

test.loc[(test.attack=='back') | (test.attack=='land') | (test.attack=='pod') | (test.attack=='neptune') |
         (test.attack=='smurf') | (test.attack=='teardrop') | (test.attack=='apache2') | (test.attack=='udpstorm') |
         (test.attack=='processtable') | (test.attack=='worm') | (test.attack=='mailbomb'),'attack_class']=1

test.loc[(test.attack=='satan') | (test.attack=='ipsweep') | (test.attack=='nmap') | (test.attack=='portsweep') |
          (test.attack=='mscan') | (test.attack=='saint'),'attack_class']=2

test.loc[(test.attack=='guess_passwd') | (test.attack=='ftp_write') | (test.attack=='imap') | (test.attack=='phf') |
          (test.attack=='multihop') | (test.attack=='warezmaster') | (test.attack=='warezclient') | (test.attack=='spy') |
          (test.attack=='xlock') | (test.attack=='xsnoop') | (test.attack=='snmpguess') | (test.attack=='snmpgetattack') |
          (test.attack=='httptunnel') | (test.attack=='sendmail') | (test.attack=='named'),'attack_class']=3

test.loc[(test.attack=='buffer_overflow') | (test.attack=='loadmodule') | (test.attack=='rootkit') | (test.attack=='perl') |
          (test.attack=='sqlattack') | (test.attack=='xterm') | (test.attack=='ps'),'attack_class']=4

# print(train.head())
# print(train.shape)

# Basic Exploratory Analysis
# Protocol type distribution
plt.figure(figsize=(8,5))
sns.countplot(x="protocol_type", data=train)
plt.savefig('model_output/Image1.png')

# service distribution
plt.figure(figsize=(8,15))
sns.countplot(y="service", data=train)
plt.savefig('model_output/Image2.png')

# flag distribution
plt.figure(figsize=(8,6))
sns.countplot(x="flag", data=train)
plt.savefig('model_output/Image3.png')

# attack distribution
plt.figure(figsize=(10,6))
sns.countplot(y="attack", data=train)
plt.savefig('model_output/Image4.png')

# attack class distribution
plt.figure(figsize=(8,6))
sns.countplot(x="attack_class", data=train)
plt.savefig('model_output/Image5.png')

# identifying relationships (between Y & numerical independent variables by comparing means)
# train.groupby('attack_class').mean().T
numeric_var_names=[key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]
cat_var_names=[key for key in dict(train.dtypes) if dict(train.dtypes)[key] in ['object', 'O']]
# print(numeric_var_names)
# print(cat_var_names)
train_num=train[numeric_var_names]
test_num=test[numeric_var_names]
# print(train_num.head(5))
train_cat=train[cat_var_names]
test_cat=test[cat_var_names]
# print(train_cat.head(5))

# Data Audit Report
# Creating Data audit Report
def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()],
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])

num_summary = train_num.apply(lambda x: var_summary(x)).T
# print(num_summary)
num_summary.to_csv('num_summary.csv1')

# Handling Outlier
def outlier_capping(x):
    x = x.clip(upper=x.quantile(0.99))
    x = x.clip(lower=x.quantile(0.01))
    return x

train_num=train_num.apply(outlier_capping)
# No missing in train dataset . So , Missing treatment not required .
def cat_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.value_counts()],
                  index=['N', 'NMISS', 'ColumnsNames'])

cat_summary=train_cat.apply(cat_summary)
# print(cat_summary)
# Dummy Variable Creation
# An utility function to create dummy variable
def create_dummies( df, colname ):
    col_dummies = pd.get_dummies(df[colname], prefix=colname, drop_first=True)
    df = pd.concat([df, col_dummies], axis=1)
    df.drop( colname, axis = 1, inplace = True )
    return(df)
#for c_feature in categorical_features
for c_feature in ['protocol_type', 'service', 'flag', 'attack']:
    train_cat = create_dummies(train_cat,c_feature)
    test_cat = create_dummies(test_cat,c_feature)
# print(train_cat.head())

# Final file for analysis
train_new = pd.concat([train_num, train_cat], axis=1)
test_new = pd.concat([test_num, test_cat], axis=1)
# print(train_new.head())
# correlation matrix (ranges from 1 to -1)
corrm = train_new.corr()
# print(corrm)
corrm.to_csv('corrm1.csv')
# visualize correlation matrix in Seaborn using a heatmap
sns.heatmap(corrm)
plt.savefig('model_output/Image6_corrm.png')

# Dropping columns based on data audit report
train_new.drop(columns=['land','wrong_fragment','urgent','num_failed_logins',"root_shell","su_attempted","num_root",
                        "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
                        'dst_host_rerror_rate','dst_host_serror_rate','dst_host_srv_rerror_rate','dst_host_srv_serror_rate',
                        'num_root','num_outbound_cmds','srv_rerror_rate','srv_serror_rate'], axis=1, inplace=True)
sns.heatmap(train_new.corr())
plt.savefig('model_output/Image7_train_new.corr.png')
# Variable reduction using Select K-Best technique
X = train_new[train_new.columns.difference(['attack_class'])]
X_new = SelectKBest(f_classif, k=15).fit(X, train_new['attack_class'] )
# print(X_new.get_support())
# print(X_new.scores_)
# capturing the important variables
KBest_features=X.columns[X_new.get_support()]
# print(KBest_features)

# Final list of variable selected for the model building using Select KBest
train=train_new
test=test_new
# Model Building
top_features=['attack_neptune','attack_normal','attack_satan','count','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_same_srv_rate','dst_host_srv_count','flag_S0','flag_SF','last_flag','logged_in','same_srv_rate','serror_rate','service_http']
X_train = train[top_features]
y_train = train['attack_class']
X_test = test[top_features]
y_test = test['attack_class']

# Building logistic Regression
# 1) LogisticRegression
lr_clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)
y_pred=lr_clf.predict(X_test)
# print(y_pred)
print("LogisticRegression Accuracy Score: ", accuracy_score(y_test, y_pred))
# 2) RidgeClassifier
rc_clf = RidgeClassifier().fit(X_train, y_train)
y_pred=rc_clf.predict(X_test)
# print(y_pred)
print("RidgeClassifier Accuracy Score: ", accuracy_score(y_test, y_pred))

# K-Nearest Neighbors
# 1) KNeighborsClassifier
k_neigh = KNeighborsClassifier(n_neighbors=3)
k_neigh.fit(X_train, y_train)
y_pred=k_neigh.predict(X_test)
# print(y_pred)
print("KNeighborsClassifier Accuracy Score: ", accuracy_score(y_test, y_pred))
# 2) NearestCentroid
nc = NearestCentroid()
nc.fit(X_train, y_train)
y_pred=nc.predict(X_test)
# print(y_pred)
print("NearestCentroid Accuracy Score: ", accuracy_score(y_test, y_pred))

# Discriminant Analysis
# 1) LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred=lda.predict(X_test)
# print(y_pred)
print("LinearDiscriminantAnalysis Accuracy Score: ", accuracy_score(y_test, y_pred))
# 2) QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred=qda.predict(X_test)
# print(y_pred)
print("QuadraticDiscriminantAnalysis Accuracy Score: ", accuracy_score(y_test, y_pred))

# Decision Trees
clf_tree = DecisionTreeClassifier( max_depth = 5)
clf_tree=clf_tree.fit( X_train, y_train )
y_pred=qda.predict(X_test)
# print(y_pred)
print("Decision Trees Accuracy Score: ", accuracy_score(y_test, y_pred))

# Fine Tuning the parameters
param_grid = {'max_depth': np.arange(3, 9),
             'max_features': np.arange(3,9)}
tree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv = 5)
tree.fit( X_train, y_train )
print(tree.best_score_)
print(tree.best_estimator_)
print(tree.best_params_)
# Building Final Decision Tree Model
clf_tree = DecisionTreeClassifier( max_depth = 8, max_features=8 )
clf_tree.fit( X_train, y_train )

# Feature Relative Importance
print(clf_tree.feature_importances_)

# summarize the selection of the attributes
feature_map = [(i, v) for i, v in itertools.zip_longest(X_train.columns, clf_tree.feature_importances_)]
print(feature_map)
Feature_importance = pd.DataFrame(feature_map, columns=['Feature', 'importance'])
Feature_importance.sort_values('importance', inplace=True, ascending=False)
print(Feature_importance)
tree_test_pred = pd.DataFrame( { 'actual':  y_test,
                            'predicted': clf_tree.predict( X_test ) } )
print(tree_test_pred.sample( n = 10 ))
print("Feature Relative Importance Accuracy Score: ", accuracy_score(tree_test_pred.actual, tree_test_pred.predicted))
tree_cm = metrics.confusion_matrix(tree_test_pred.actual, tree_test_pred.predicted, labels=[1,0])
sns.heatmap(tree_cm, annot=True, fmt='.2f', xticklabels=["Left", "No Left"], yticklabels=["Left", "No Left"])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('model_output/Image8_Decision Trees_confusion_matrix.png')

# Naive Bayes Model
# 1) BernoulliNB
bnb_clf = BernoulliNB()
bnb_clf.fit(X_train, y_train)
y_pred=bnb_clf.predict(X_test)
# print(y_pred)
print("BernoulliNB Accuracy Score: ", accuracy_score(y_test, y_pred))
nb_cm = metrics.confusion_matrix(y_test,y_pred)
sns.heatmap(nb_cm, annot=True,  fmt='.2f', xticklabels=["no", "Yes"], yticklabels=["No", "Yes"])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('model_output/Image9_BernoulliNB_confusion_matrix.png')

# 2) GaussianNB
gnb_clf = GaussianNB()
gnb_clf.fit(X_train, y_train)
y_pred=gnb_clf.predict(X_test)
# print(y_pred)
print("GaussianNB Accuracy Score: ", accuracy_score(y_test, y_pred))
nb_cm = metrics.confusion_matrix( y_test, y_pred )
sns.heatmap(nb_cm, annot=True,  fmt='.2f', xticklabels = ["no", "Yes"] , yticklabels = ["No", "Yes"])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('model_output/Image9_GaussianNB_confusion_matrix.png')

# Support Vector Machine (SVM)
# 1) LinearSVC
svm_clf = LinearSVC(random_state=0, tol=1e-5)
svm_clf.fit(X_train, y_train)
y_pred=svm_clf.predict(X_test)
# print(y_pred)
print("LinearSVC Accuracy Score: ", accuracy_score( y_test, y_pred ))

# 2) SVC
model = SVC(kernel='rbf', class_weight='balanced',gamma='scale')
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
# print(y_pred)
print("SVC Accuracy Score: ", accuracy_score( y_test, y_pred ))

# Stochastic Gradient Descent (SGD)
model = SGDClassifier(loss="hinge", penalty="l2")
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
# print(y_pred)
print("Stochastic Gradient Descent Accuracy Score: ", accuracy_score( y_test, y_pred ))
n_iters = [5, 10, 20, 50, 100, 1000]
scores = []
for n_iter in n_iters:
    model = SGDClassifier(loss="hinge", penalty="l2", max_iter=n_iter)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

plt.title("Effect of n_iter")
plt.xlabel("n_iter")
plt.ylabel("score")
plt.plot(n_iters, scores)
plt.savefig('model_output/Image10_Stochastic Gradient Descent_graph.png')
# losses
losses = ["hinge", "log", "modified_huber", "perceptron", "squared_hinge"]
scores = []
for loss in losses:
    model = SGDClassifier(loss=loss, penalty="l2", max_iter=1000)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

plt.xlabel("loss")
plt.ylabel("score")
plt.title("Effect of loss")
x = np.arange(len(losses))
plt.xticks(x, losses)
plt.plot(x, scores)
plt.savefig('model_output/Image11_Stochastic Gradient Descent_losses_graph.png')
params = {
    "loss" : ["hinge", "log", "squared_hinge", "modified_huber"],
    "alpha" : [0.0001, 0.001, 0.01, 0.1],
    "penalty" : ["l2", "l1", "none"],
}

model = SGDClassifier(max_iter=100)
clf = GridSearchCV(model, param_grid=params)
clf.fit(X_train, y_train)
print(clf.best_score_)
y_pred=clf.predict(X_test)
# print(y_pred)
print("SGDClassifier losses Accuracy Score: ",accuracy_score( y_test, y_pred ))

# Neural Network Model
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
# Now apply the transformations to the data:
train_X = scaler.transform(X_train)
test_X = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(train_X,y_train)
y_pred=mlp.predict(test_X)
# print(y_pred)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(mlp.coefs_)
print("Neural Network Model Accuracy Score: ", accuracy_score( y_test, y_pred ))

# Bagging Algorithms
# 1. Bagged Decision Trees
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())
# print(model.fit(X_train, y_train))
y_pred=model.predict(X_test)
# print(y_pred)
print("Bagged Decision Trees Accuracy Score: ", accuracy_score( y_test, y_pred ))

# 2. Random Forest
seed = 7
num_trees = 100
max_features = 3
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())
# print(model.fit(X_train, y_train))
y_pred=model.predict(X_test)
# print(y_pred)
print("Random Forest Accuracy Score: ", accuracy_score( y_test, y_pred ))

# 3. Extra Trees
seed = 7
num_trees = 100
max_features = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())
# print(model.fit(X_train, y_train))
y_pred=model.predict(X_test)
# print(y_pred)
print("Extra Trees Accuracy Score: ", accuracy_score( y_test, y_pred ))

# Boosting Algorithms
# 1. AdaBoost
seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())
# print(model.fit(X_train, y_train))
y_pred=model.predict(X_test)
# print(y_pred)
print("AdaBoost Accuracy Score: ", accuracy_score( y_test, y_pred ))

# 2. Stochastic Gradient Boosting
seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())
# print(model.fit(X_train, y_train))
y_pred=model.predict(X_test)
# print(y_pred)
print("Stochastic Gradient Boosting Accuracy Score: ", accuracy_score( y_test, y_pred ))

# Voting Ensemble
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X_train, y_train, cv=kfold)
print(results.mean())
# print(ensemble.fit(X_train, y_train))
y_pred=ensemble.predict(X_test)
# print(y_pred)
print("Voting Ensemble Accuracy Score: ", accuracy_score( y_test, y_pred ))

# Save Model
# Saving model to disk of random forest
pickle.dump(lr_clf, open('model1.pkl','wb'))
model=pickle.load(open('model1.pkl', 'rb'))
model.predict([[1,0,0,229,0.06,0.00,0.04,10,0,0,21,0,0.04,0.00,0]])
plt.show()