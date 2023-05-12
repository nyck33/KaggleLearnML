import numpy as np
import pandas as pd
# import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as tts
import os


#############################################################################
#from task5.py


############################################################################

def train_test_split(dataset: pd.DataFrame,
                     target_col: str,
                     test_size: float,
                     stratify: bool,
                     random_state: int):#-> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # TODO: Write the necessary code to split a dataframe into a Train and Test feature dataframe and a Train and Test
    X = dataset.copy() #dataset.drop([target_col], axis=1).copy()
    del X[target_col]
    #X = X.to_numpy().reshape(X.shape[0], X.shape[1])
    y = dataset[target_col].copy()
    #y = y.set_axis(X.index)
    #frame = {target_col: y}
    #y = pd.DataFrame(frame, index=dataset.index)
    #y = np.array(y)
    if stratify:
        X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, stratify=y, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=random_state)

    #XCols = X.columns.values.tolist()

    #train_features = pd.DataFrame(X_train, XCols, index=dataset.index)
    #test_features = pd.DataFrame(X_test, XCols, index=dataset.index)


    return (X_train, X_test,y_train, y_test)


class PreprocessDataset:
    def __init__(self,
                 train_features: pd.DataFrame,
                 test_features: pd.DataFrame,
                 one_hot_encode_cols: list[str],
                 min_max_scale_cols: list[str],
                 n_components: int,
                 feature_engineering_functions: dict #{'name": func}
                 ):
        # TODO: Add any state variables you may need to make your functions work
        self.trainFeatures = train_features
        self.testFeatures = test_features
        self.oneHotEncodeCols = one_hot_encode_cols
        self.minMaxScaleCols = min_max_scale_cols
        self.nComponents = n_components
        self.featureEngineeringFunctions = feature_engineering_functions

        # compound transform so check if this None, this should be passed in train methods
        self.partialyTransformedTrainDf = None
        self.partialyTransformedTestDf = None
        ##############ohe
        self.ohe = None
        # the vals found in col during training, see if test feature is in this list
        self.colNameToColValsD = {}
        # col name before one hot encoding: name after
        self.colNameToNewColNamesD = {}
        # the ohe df after train
        self.oheTrainedDf = None
        # the test ohe df after test
        self.oheTestedDf = None
        ########### minmax
        self.minmaxTrainedDf = None
        self.scaler = None
        #### pca
        self.pca = None
        self.pcaTrainedDf = None
        self.pcaTrainDroppedCols = None
        self.pcaTrainKeptCols = None

    def one_hot_encode_columns_train(self): #-> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with the categorical column names in
        # the variable one_hot_encode_cols "one hot" encoded
        '''
        if self.partialyTransformedTrainDf is None:
            one_hot_encoded_dataset = self.trainFeatures.copy()
        else:
            one_hot_encoded_dataset = self.partialyTransformedTrainDf.copy()
        '''
        one_hot_encoded_dataset = self.trainFeatures.copy()
        ohe = OneHotEncoder(handle_unknown='ignore')
        self.ohe = ohe

        colsToTransformArr = one_hot_encoded_dataset[self.oneHotEncodeCols].to_numpy().tolist() #.to_numpy().reshape(-1,1)
        fitted = ohe.fit(colsToTransformArr)
        #colsToTransformArr = np.array(colsToTransformArr)
        transformed = ohe.transform(colsToTransformArr)
        transformed = transformed.toarray().tolist()
        newColNames = ohe.get_feature_names_out(self.oneHotEncodeCols)
        onehotDf = pd.DataFrame(transformed, columns=newColNames, index=one_hot_encoded_dataset.index)
        # add columns one by one as in test
        trainCols = one_hot_encoded_dataset.columns.tolist()
        onehotCols = onehotDf.columns.tolist()
        trainOnlyCols = [x for x in trainCols if x not in onehotCols]

        for col in trainOnlyCols:
            onehotDf[col] = one_hot_encoded_dataset.loc[:, col]
        for col in self.oneHotEncodeCols:
            del onehotDf[col]

        self.oheTrainedDf = onehotDf
        self.partialyTransformedTrainDf = onehotDf

        return onehotDf


    def one_hot_encode_columns_test(self): #-> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with the categorical column names in
        # the variable one_hot_encode_cols "one hot" encoded
        '''
        Iterate the test features df by row, iterate the one hot encode cols and if there's a match then check the train features to see which col needs a 1
        if none match then all 0's
        :param self:
        :return:
        '''
        # todo: check if this is testFeatures or some df saved during train
        # TODO: Write the necessary code to create a dataframe with the categorical column names in
        # the variable one_hot_encode_cols "one hot" encoded
        testDf = self.testFeatures.copy()
        trainDf = self.oheTrainedDf.copy()
        ohe = self.ohe

        colsToTransformArr = testDf[self.oneHotEncodeCols].to_numpy().tolist()  # .to_numpy().reshape(-1,1)
        transformed = ohe.transform(colsToTransformArr)
        transformed = transformed.toarray().tolist()
        newColNames = ohe.get_feature_names_out(self.oneHotEncodeCols)
        onehotDf = pd.DataFrame(transformed, columns=newColNames, index=testDf.index)

        # here now I have a DF with 2 records and possibly new cols that don't exist
        # I need the trainDf and then add records
        trainCols = trainDf.columns.tolist()
        onehotCols = onehotDf.columns.tolist()
        trainOnlyCols = [x for x in trainCols if x not in onehotCols]

        #appending the trainOnlyCols to onehotDf
        for col in trainOnlyCols:
            onehotDf[col] = testDf.loc[:, col]

        self.oheTestedDf = onehotDf
        self.partialyTransformedTestDf = onehotDf
        return onehotDf

    def min_max_scaled_columns_train(self): #-> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with the numerical column names in the variable min_max_scale_cols scaled to the min and max of each column
        min_max_scaled_dataset = self.trainFeatures.copy()
        #todo: check if self.partiallyTrained is None

        scaler = MinMaxScaler()
        colsToScaleArr = min_max_scaled_dataset[self.minMaxScaleCols].to_numpy().tolist()
        fitted = scaler.fit(colsToScaleArr)
        self.scaler = scaler

        transformed = scaler.transform(colsToScaleArr)

        minmaxDf = pd.DataFrame(transformed, columns=self.minMaxScaleCols, index=min_max_scaled_dataset.index)

        unscaledCols = min_max_scaled_dataset.columns.tolist()
        scaledCols = self.minMaxScaleCols

        needToAddCols = [x for x in unscaledCols if x not in scaledCols]
        for col in needToAddCols:
            minmaxDf[col] = min_max_scaled_dataset.loc[:, col]

        self.minmaxTrainDf = minmaxDf
        return minmaxDf


    def min_max_scaled_columns_test(self): #-> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with the numerical column names in the variable min_max_scale_cols scaled to the min and max of each column
        '''
        take in test features and adjust values to fit with scale in same column of the train data

        :return:
        '''
        scaler = self.scaler
        min_max_scaled_dataset = self.testFeatures.copy()
        #todo: check for partiallyTrained
        minmaxCols = self.minMaxScaleCols
        data = min_max_scaled_dataset[minmaxCols].to_numpy().tolist()
        #fitted = scaler.fit(data)
        scaleTransformed = scaler.transform(data).tolist()

        minmaxDf = pd.DataFrame(scaleTransformed, columns=minmaxCols, index=min_max_scaled_dataset.index)

        unscaledCols = min_max_scaled_dataset.columns.tolist()
        scaledCols = self.minMaxScaleCols

        needToAddCols = [x for x in unscaledCols if x not in scaledCols]
        for col in needToAddCols:
            minmaxDf[col] = min_max_scaled_dataset.loc[:, col]

        return minmaxDf

    def pca_train(self): #-> pd.DataFrame:
        # TODO: use PCA to reduce the train_df to n_components principal components
        # Name your new columns component_1, component_2 .. component_n
        # 1. init PCA with random seed of 0 and n_components
        # 2.  train PCA on training features dropping any cols that have NA values
        # 3.  transform training set using pca
        # 4.  create a DF with col names component_1, component_2, ...n for each component created

        #oneHotEncodedDf = self.one_hot_encode_columns_train()
        #gets back one hot encoded and scaled df
        #self.trainFeatures = oneHotEncodedDf.copy()
        #oheAndScaledDf = self.min_max_scaled_columns_train()
        #pca_dataset = oheAndScaledDf.copy()
        trainDf = self.trainFeatures.copy()

        #pca_dataset = trainDf.select_dtypes(include=[np.number])
        for col in trainDf.columns.tolist():
            if not np.issubdtype(trainDf[col].dtype, np.number):
                del trainDf[col]
                if self.pcaTrainDroppedCols is None:
                    self.pcaTrainDroppedCols = [col]
                else:
                    self.pcaTrainDroppedCols.append(col)

        # delete any columns with NA
        beforeDropnaCols = trainDf.columns.tolist()
        noNADf = trainDf.dropna(axis=1, thresh=1, inplace=False)
        # compare cols and add them for test
        afterDropnaCols = noNADf.columns.tolist()
        dropnaedCols = [x for x in beforeDropnaCols if x not in afterDropnaCols]
        if len(dropnaedCols) > 0:
            for c in dropnaedCols:
                if self.pcaTrainDroppedCols is None:
                    self.pcaTrainDroppedCols = [c]
                else:
                    self.pcaTrainDroppedCols.append(c)


        pca_dataset = noNADf.copy()
        pca = PCA(random_state=0, n_components=self.nComponents) #svd_solver='arpack')
        # make column names
        colNames = []
        prefix = "component_"
        for i in range(self.nComponents):
            colName = prefix + str(i+1)
            colNames.append(colName)
        data = pca_dataset.to_numpy().tolist()
        #data = pca_dataset.to_numpy().reshape(-1,1).tolist()
        fitted = pca.fit(data)
        self.pca = pca

        transformed = pca.transform(data)
        newDf = pd.DataFrame(transformed, columns=colNames) #index=pca_dataset.index)

        return newDf

    def pca_test(self): #-> pd.DataFrame:
        # TODO: use PCA to reduce the test_df to n_components principal components
        # Name your new columns component_1, component_2 .. component_n
        testDf = self.testFeatures.copy()

        droppedCols = self.pcaTrainDroppedCols
        for col in droppedCols:
            del testDf[col]

        # delete any columns with NA
        beforeDropnaCols = testDf.columns.tolist()
        noNADf = testDf.dropna(axis=1, thresh=1, inplace=False)
        afterDropnaCols = noNADf.columns.tolist()
        dropnaedCols = [x for x in beforeDropnaCols if x not in afterDropnaCols]

        #assert len(dropnaedCols) == 0, f"pca test dropped NA cols: {dropnaedCols}"

        pca_dataset = noNADf.copy()
        pca = self.pca
        # make column names
        colNames = []
        prefix = "component_"
        for i in range(self.nComponents):
            colName = prefix + str(i + 1)
            colNames.append(colName)
        data = pca_dataset.to_numpy().tolist()
        # data = pca_dataset.to_numpy().reshape(-1,1).tolist()


        transformed = pca.transform(data)
        newDf = pd.DataFrame(transformed, columns=colNames)  # index=pca_dataset.index)

        return newDf

    def feature_engineering_train(self):  #-> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with feature engineering functions applied
        # from the feature_engineering_functions dict (the dict format is {'feature_name':function,})
        # each feature engineering function will take in type pd.DataFrame and return a pd.Series

        feature_engineered_dataset = self.trainFeatures.copy()

        feDict = self.featureEngineeringFunctions

        for name, f in feDict.items():
            # create a column in the training dataframe
            resSeries = feDict[name](feature_engineered_dataset)
            feature_engineered_dataset[name] = resSeries

        return feature_engineered_dataset

    def feature_engineering_test(self): #-> pd.DataFrame:
        # TODO: Write the necessary code to create a dataframe with feature engineering functions applied
        # from the feature_engineering_functions dict (the dict format is {'feature_name':function,})
        # each feature engineering function will take in type pd.DataFrame and return a pd.Series
        feature_engineered_dataset = self.testFeatures.copy()

        feDict = self.featureEngineeringFunctions

        for name, f in feDict.items():
            # create a column in the training dataframe
            resSeries = feDict[name](feature_engineered_dataset)
            feature_engineered_dataset[name] = resSeries

        return feature_engineered_dataset

    def preprocess(self):  #-> tuple[pd.DataFrame, pd.DataFrame]:
        # TODO: Use the functions you wrote above to create train/test splits of the features and target with scaled and encoded values
        # for the columns specified in the init function
        train_features = self.trainFeatures.copy()
        # train the features first then run tests
        oheDF = self.one_hot_encode_columns_train()
        self.trainFeatures = oheDF
        oheAndMinmaxDf = self.min_max_scaled_columns_train()
        self.trainFeatures = oheAndMinmaxDf
        oheMinmaxFeTrainDf = self.feature_engineering_train()
        self.trainFeatures = oheMinmaxFeTrainDf
        #### traing complete now test
        test_features = self.trainFeatures.copy()
        oheTestedDf = self.one_hot_encode_columns_test()
        self.testFeatures = oheTestedDf
        oheMinmaxTestedDf = self.min_max_scaled_columns_test()
        self.testFeatures = oheMinmaxTestedDf
        oheMinmaxFeTestedDf = self.feature_engineering_test()
        self.testFeatures = oheMinmaxFeTestedDf
        return oheMinmaxFeTrainDf, oheMinmaxFeTestedDf