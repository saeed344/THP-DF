import numpy as np
import pandas as pd
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, auc
# from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.preprocessing import scale
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import utils.tools as utils
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import scale, StandardScaler
from sklearn.preprocessing import normalize, Normalizer
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
from sklearn.decomposition import PCA, NMF, KernelPCA, SparsePCA, FastICA, PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoLarsCV, LassoLars
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.preprocessing import scale, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize, Normalizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import LassoLarsCV, LassoLars
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import scale, StandardScaler
from sklearn.preprocessing import normalize, Normalizer
from sklearn.preprocessing import minmax_scale, MinMaxScaler
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler
from sklearn.decomposition import PCA, NMF, KernelPCA, SparsePCA, FastICA, PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoLarsCV, LassoLars
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.manifold import SpectralEmbedding
from sklearn.ensemble import ExtraTreesClassifier
"""""

def zscore_scaler(data):
    data = scale(H)
    return data


def normalizer(data):
    data = Normalizer().fit_transform(data)
    return data


def minmaxscaler(data):
    data = MinMaxScaler().fit_transform(data)
    return data


def maxabsscaler(data):
    data = MaxAbsScaler().fit_transform(data)
    return data


def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)  # axis represents the obtained mean value by column
    stdVal = np.std(dataMat, axis=0)
    newData = dataMat - meanVal
    new_data = newData / stdVal
    return new_data, meanVal


def covArray(dataMat):
    # obtain the  covariance matrix
    covMat = np.cov(dataMat, rowvar=0)
    return covMat


def featureMatrix(dataMat):
    eigVals, eigVects = np.linalg.eig(np.mat(dataMat))
    # datermine the eigenvalue and eigenvector
    return eigVals, eigVects


def percentage2n(eigVals, percentage=0.99):
    # percentage represents the rate of contribution
    sortArray = np.sort(eigVals)  # ascending sort
    sortArray = sortArray[-1::-1]  # descending order
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num


def factorAnalysis(data, percentage=0.9):
    dataMat = np.array(data)
    newData, meanVal = zeroMean(data)  # equalization
    covMat = covArray(newData)  # covariance matrix
    eigVals, eigVects = featureMatrix(covMat)
    n_components = percentage2n(eigVals, percentage)
    clf = FactorAnalysis(n_components=n_components)
    new_data = clf.fit_transform(dataMat)
    return new_data


def pca(data, percentage=0.9):
    dataMat = np.array(data)
    newData, meanVal = zeroMean(data)  # equalization
    covMat = covArray(newData)  # covariance matrix
    eigVals, eigVects = featureMatrix(covMat)
    n_components = percentage2n(eigVals, percentage)
    clf = PCA(n_components=n_components)
    new_data = clf.fit_transform(dataMat)
    return new_data


def kernelPCA(data, percentage=0.2):
    dataMat = np.array(data)
    newData, meanVal = zeroMean(data)
    covMat = covArray(newData)
    eigVals, eigVects = featureMatrix(covMat)
    n_components = percentage2n(eigVals, percentage)
    # n_component=n_components
    clf = KernelPCA(n_components=n_components, kernel='rbf', gamma=1 / 1318)  # rbf linear poly
    new_data = clf.fit_transform(dataMat)
    return new_data


def mutual_mutual(data, label, k=10):  # 互信息
    model_mutual = SelectKBest(mutual_info_classif, k=k)
    new_data = model_mutual.fit_transform(data, label)
    mask = model_mutual.get_support(indices=True)
    return new_data, mask


def lassodimension(data, label, alpha=np.array([0.005])):
    # def lassodimension(data,label,alpha=np.array([0.005])):####acc=91.82%
    # def lassodimension(data,label,alpha=np.array([0.01,0.05,0.1])):#The alpha value range is 0.001, 0.002, 0.005, 0.01, 0.015, 0.02
    lassocv = LassoCV(cv=5, alphas=alpha).fit(data, label)
    x_lasso = lassocv.fit(data, label)  # Substituting alpha for dimensionality reduction
    mask = x_lasso.coef_ != 0
    new_data = data[:, mask]
    return new_data, mask


def logistic_dimension(data, label, parameter=1):
    logistic_ = LogisticRegression(penalty="l1", C=parameter, max_iter=30)
    model = SelectFromModel(logistic_)
    new_data = model.fit_transform(data, label)
    mask = model.get_support(indices=True)
    return new_data, mask


# using factor analysis to reduce the dimension
def elasticNet(data, label, alpha=np.array([0.01])):
    enetCV = ElasticNetCV(alphas=alpha, l1_ratio=0.01).fit(data, label)
    enet = ElasticNet(alpha=enetCV.alpha_, l1_ratio=0.01)
    enet.fit(data, label)
    mask = enet.coef_ != 0
    new_data = data[:, mask]
    return new_data, mask, enetCV.alpha_


# def elasticNet(data,label,alpha =np.array([0.01])):
#    enetCV = ElasticNetCV(alphas=alpha,l1_ratio=0.01).fit(data,label)
#    enet=ElasticNet(alpha=enetCV.alpha_, l1_ratio=0.01)
#    enet.fit(data,label)
#    mask = enet.coef_ != 0
#    new_data = data[:,mask]
#    return new_data,mask,enetCV.alpha_

def SE(data, n_components=20):
    embedding = SpectralEmbedding(n_components=n_components)
    X_transformed = embedding.fit_transform(data)
    return X_transformed


def LLE(data, n_components=30):
    embedding = LocallyLinearEmbedding(n_components=n_components)
    X_transformed = embedding.fit_transform(data)
    return X_transformed


def TSVD(data, n_components=300):
    svd = TruncatedSVD(n_components=n_components)
    new_data = svd.fit_transform(data)
    return new_data


###############################################################################################################3

data_train = sio.loadmat('EBGW4_cross.mat')
data = data_train.get('shuEBGW4_cross')  # Remove the data in the dictionary
shu = data
shu = scale(shu)
# label1=np.ones((284,1))#Value can be changed
# label2=np.zeros((303,1))
label1 = np.ones((1420, 1))  # Value can be changed
label2 = np.zeros((1514, 1))
label = np.append(label1, label2)
#####################################################################################
# data_1,mask,H=elasticNet(shu,label)#引用降维'

data_1, mask = lassodimension(shu, label)

# data_1=SE(shu)
#####################################################
X = data_1
y = label  #############降维
#########################################################################################
# X=shu
# y=label##############不降维

####################################
sepscores = []
ytest = np.ones((1, 2)) * 0.5
yscore = np.ones((1, 2)) * 0.5
"""""
data_=pd.read_csv(r'AAC(main 90).csv')
data1=np.array(data_)
data2=data1[1:,1:]
#############################################################
label1 = np.ones((176, 1))  # Value can be changed
label2 = np.zeros((433, 1))
label = np.append(label1, label2)
scale_data = scale(data2)
MinMax = MinMaxScaler(feature_range=((0, 1)))
MinMaxdata = MinMax.fit_transform(data2)
X=scale_data
y=label
sepscores = []
ytest = np.ones((1, 2)) * 0.5
yscore = np.ones((1, 2)) * 0.5
################################################################################################
cv_clf = SVC(probability=True)  # using support vector machine

####################################################################################################3
skf = StratifiedKFold(n_splits=5)
for train, test in skf.split(X, y):
    y_train = utils.to_categorical(y[train])
    hist = cv_clf.fit(X[train], y[train])
    y_score = cv_clf.predict_proba(X[test])
    yscore = np.vstack((yscore, y_score))
    y_test = utils.to_categorical(y[test])
    ytest = np.vstack((ytest, y_test))
    fpr, tpr, _ = roc_curve(y_test[:, 0], y_score[:, 0])
    roc_auc = auc(fpr, tpr)
    y_class = utils.categorical_probas_to_classes(y_score)
    y_test_tmp = y[test]
    acc, precision, npv, sensitivity, specificity, mcc, f1 = utils.calculate_performace(len(y_class), y_class,
                                                                                        y_test_tmp)
    sepscores.append([acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc])
    print('SVC:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
          % (acc, precision, npv, sensitivity, specificity, mcc, f1, roc_auc))
scores = np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0] * 100, np.std(scores, axis=0)[0] * 100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1] * 100, np.std(scores, axis=0)[1] * 100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2] * 100, np.std(scores, axis=0)[2] * 100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3] * 100, np.std(scores, axis=0)[3] * 100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4] * 100, np.std(scores, axis=0)[4] * 100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5] * 100, np.std(scores, axis=0)[5] * 100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6] * 100, np.std(scores, axis=0)[6] * 100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7] * 100, np.std(scores, axis=0)[7] * 100))

result1 = np.mean(scores, axis=0)
H1 = result1.tolist()
sepscores.append(H1)
result = sepscores
row = yscore.shape[0]
yscore = yscore[np.array(range(1, row)), :]
yscore_sum = pd.DataFrame(data=yscore)
yscore_sum.to_csv('yscore_EBGW4_cross.csv')
ytest = ytest[np.array(range(1, row)), :]
ytest_sum = pd.DataFrame(data=ytest)
ytest_sum.to_csv('ytest_EBGW4_cross.csv')
fpr, tpr, _ = roc_curve(ytest[:, 0], yscore[:, 0])
auc_score = np.mean(scores, axis=0)[7]
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='EBGW4_cross ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
data_csv = pd.DataFrame(data=result)
data_csv.to_csv('result_EBGW4_cross.csv')


