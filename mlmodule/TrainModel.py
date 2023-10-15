import os
import numpy as np
import pandas as pd
import joblib
import copy
import optuna
import random
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'# 關閉tensorflow訊息
import tensorflow as tf
from src.preprocess_method import DWT_denoise
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from keras.models import Sequential
from keras.losses import BinaryCrossentropy
from keras.layers import Dense, Input, Lambda, Reshape, Activation, Dropout
from keras.optimizers import Adam, Adagrad, RMSprop, Adadelta 

optuna.logging.disable_default_handler()# 關閉optuna FineTune過程


class Model_train:

    def __init__(self, user_id: str):
        self.user_id = user_id


    """ 儲存模型 """
    def __save_model(self, model, model_name: str):
        save_path = f"./model/{self.user_id}/{model_name}.pkl"
        
        try:
            joblib.dump(model, save_path)
        except FileNotFoundError:
            os.makedirs(f"./model/{self.user_id}")
            joblib.dump(model, save_path)


    """ 資料處裡 """
    def __data_preprocess(self, data):
        
        trans_data = pd.DataFrame(data)
        trans_data = trans_data/1000# 瓦轉千瓦
        trans_data = np.round(trans_data, 2)# 四捨五入至第二位

        return trans_data


    """ 小波轉換 """
    def __wavelet_transform(self, data):
        tmp_data = copy.deepcopy(data)
        de = DWT_denoise()
        
        ## 小波轉換
        # date = data.iloc[:,0]# 保留原有日期
        tmp_data.drop(["Report_time"], axis=1, inplace=True)# 刪除日期
        dwt_data = de.transform(tmp_data.values, 'db4', level=1)
        dwt_data = self.__data_preprocess(dwt_data)

        ## 裝上日期
        # columns = dwt_data.columns.tolist()
        # columns.insert(0, 'Report_time')
        # dwt_data= pd.DataFrame(dwt_data, columns=columns)
        # dwt_data['Report_time'] = date

        return  dwt_data


    """ PCA特徵縮放 """
    def __pca_preprocess(self, data, compression_dim):
        ## PCA轉換
        pca_model = PCA(n_components=compression_dim)
        pca_model.fit(data.values)
        pca_data = pca_model.transform(data.values)
        pca_data = pd.DataFrame(data=pca_data)

        ## 儲存PCA模型
        self.__save_model(model=pca_model, model_name=f"PCA_model_{self.user_id}")

        return pca_data
    

    """ 提取分群後資料 """
    def __get_ClusterData(self, data, data_label):
        tmp_data = copy.deepcopy(data)
        tmp_data["cluster_label"] = data_label
        
        ## 獲取第一群資料   
        data_mask1 = tmp_data["cluster_label"] == 0
        data_cluster1 = tmp_data[data_mask1]
        data_cluster1 = data_cluster1.drop(["cluster_label"], axis=1)
        data_cluster1 = data_cluster1.reset_index(drop=True)

        ## 獲取第二群資料
        data_mask2 = tmp_data["cluster_label"] == 1
        data_cluster2 = tmp_data[data_mask2]
        data_cluster2 = data_cluster2.drop(["cluster_label"], axis=1)
        data_cluster2 = data_cluster2.reset_index(drop=True)

        return data_cluster1, data_cluster2


    """ Kmeans分群 """
    def __kmeans_cluster(self, data_H, data_L):
        ## Kmeans model train＆predict 使用高階特徵進行訓練
        kmeans_model = KMeans(
            n_clusters=2,
            init='k-means++',
            max_iter=10000,
            n_init=10,
            tol=1e-04).fit(data_H)
        kmeans_label = kmeans_model.predict(data_H)

        data_cluster1_H, data_cluster2_H = self.__get_ClusterData(data_H, kmeans_label)# 取出各群資料(高階)
        data_cluster1_L, data_cluster2_L = self.__get_ClusterData(data_L, kmeans_label)# 取出各群資料低階)

        ## 儲存模型
        self.__save_model(model=kmeans_model, model_name=f"kmeans_model_{self.user_id}")

        return data_cluster1_H, data_cluster2_H, data_cluster1_L, data_cluster2_L


    """ 提取過濾後資料 """
    def __get_FilterData(self, data, data_label):
        tmp_data = copy.deepcopy(data)
        tmp_data["outlier_label"] = data_label
        
        ## 獲取無雜訊資料 
        clean_mask = tmp_data["outlier_label"] == 1         
        clean_data = tmp_data[clean_mask]
        clean_data = clean_data.drop(["outlier_label"], axis=1)
        clean_data = clean_data.reset_index(drop=True)

        ## 獲取outlier資料
        outlier_mask = tmp_data["outlier_label"] == -1
        outlier_data = tmp_data[outlier_mask]
        outlier_data = outlier_data.drop(["outlier_label"], axis=1)
        outlier_data = outlier_data.reset_index(drop=True)

        return clean_data, outlier_data


    """ 雜訊用電事件過濾 """
    def __outlier_detecion(self, data_H, data_L):
        ## LocalOurlierFacter model train＆predict
        LOF_model = LocalOutlierFactor(n_neighbors=20, contamination=float(0.047))# 1 normal, -1 abnormal
        outlier_label = LOF_model.fit_predict(data_H)

        clean_data_H, outlier_data_H = self.__get_FilterData(data_H, outlier_label)# 取出雜訊資料與無雜訊用電資料(高階)
        clean_data_L, outlier_data_L = self.__get_FilterData(data_L, outlier_label)# 取出雜訊資料與無雜訊用電資料(低階)

        ## 儲存模型
        self.__save_model(model=LOF_model, model_name=f"LOF_model_{self.user_id}")

        return  clean_data_H, outlier_data_H, clean_data_L, outlier_data_L


    """ 切割資料 """
    def __data_split(self, data_H, data_L):
        ## 資料集拆成 80%訓練集, 20%驗證集
        train_data_H, val_data_H, train_data_L, val_data_L = train_test_split(data_H, data_L, train_size=0.8)

        return train_data_H, val_data_H, train_data_L, val_data_L

    """ 組合資料 """
    def __data_concat(self, normal_data, abnormal_data):
        new_data = pd.concat([normal_data, abnormal_data], axis=0)

        return new_data

    """ 標記資料 """
    def __label_data(self, clean_data, outlier_data):
        normal_data = copy.deepcopy(clean_data)
        abnormal_data = copy.deepcopy(outlier_data)
        normal_data["label"] = 0# 正常資料標記0
        abnormal_data["label"] = 1# 異常資料標記1

        return normal_data, abnormal_data

    """ 製作測試資料 """
    def __make_data(self, clean_data_H, outlier_data_H, clean_data_L, outlier_data_L):
        ## 資料標記
        normal_data_H, abnormal_data_H = self.__label_data(clean_data_H, outlier_data_H)# 高階特徵
        normal_data_L, abnormal_data_L = self.__label_data(clean_data_L, outlier_data_L)# 低階特徵

        train_data_H, val_data_H, train_data_L, val_data_L = self.__data_split(normal_data_H, normal_data_L)# 將正常資料集拆成8:2
        
        val_data_H = self.__data_concat(val_data_H, abnormal_data_H)# 異常資料混入驗證集(高階)
        val_data_L = self.__data_concat(val_data_L, abnormal_data_L)# 異常資料混入驗證集(低階)

        val_data_H, val_data_L = shuffle(val_data_H, val_data_L)# 資料打亂

        
        return  train_data_H, val_data_H, train_data_L, val_data_L

    """ 取出資料label """
    def __get_label(self, data):
        new_data = copy.deepcopy(data)
        data_label = new_data["label"]
        new_data = new_data.drop(["label"], axis=1)

        return new_data, data_label


    """ 訓練OCSVM模型 """
    def __train_OCSVM(self, train_data_value, train_data_label, val_data_value, val_data_label, group):
        ## 訂出要找的範圍
        def objective(trial):
            params = {
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
                "gamma": trial.suggest_float("gamma", 1e-3, 2, log=True),
                "tol":trial.suggest_float("tol", 1e-6, 1e-1, log=True),
                "nu": 0.005
            }
    
            OCSVM = OneClassSVM(**params)
            OCSVM.fit(train_data_value, train_data_label)
            OCSVM_predict = OCSVM.predict(val_data_value)
        
            for i in range(len(OCSVM_predict)):
                if OCSVM_predict[i] == 1:
                    OCSVM_predict[i] = 0
                else:
                    OCSVM_predict[i] = 1
    
            return f1_score(val_data_label, OCSVM_predict)
        
        ## 最佳參數尋找
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)

        ## 計算FineTune後模型各項效能
        def detailed_objective(trial):
            params = {
                "nu": 0.005,
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
                "gamma": trial.suggest_float("gamma", 1e-3, 2, log=True),
                "tol": trial.suggest_float("tol", 1e-6, 1e-1, log=True)
            }

            OCSVM = OneClassSVM(**params)
            OCSVM.fit(train_data_value, train_data_label)
            OCSVM_predict = OCSVM.predict(val_data_value)
            
            for i in range(len(OCSVM_predict)):
                if OCSVM_predict[i] == 1:
                    OCSVM_predict[i] = 0
                else:
                    OCSVM_predict[i] = 1

            acc = accuracy_score(val_data_label, OCSVM_predict)
            recall = recall_score(val_data_label, OCSVM_predict)
            precision = precision_score(val_data_label, OCSVM_predict)
            f1 = f1_score(val_data_label, OCSVM_predict)

            return acc, precision, recall, f1

        ## 印出最佳參數各項效能
        print(f"OCSVM_group{group}_score: ")
        score_name = ["Accuracy_socre", "Precision_score", "Recall_score", "F1_score"]
        score = ["%.2f" % elem for elem in detailed_objective(study.best_trial)]# 格式化至小數第二位
        score = [float(i) for i in score]# str to float
        print(dict(zip(score_name, score)))
        print()
        #print('Best trial parameters:', study.best_trial.params)


        ## 取最佳參數進行訓練
        best_OCSVM_model = OneClassSVM(
            nu=0.005,
            kernel=study.best_trial.params["kernel"],
            gamma=study.best_trial.params["gamma"],
            tol=study.best_trial.params["tol"]
        )                                        
        best_OCSVM_model.fit(train_data_value)

        ## 儲存模型
        self.__save_model(model=best_OCSVM_model, model_name=f"OCSVM_model_group{group}_{self.user_id}")
    
        return


    """ 訓練IsolationForest模型 """
    def __train_ILF(self, train_data_value, train_data_label, val_data_value, val_data_label, group):
        ## 訂出要找的範圍
        def objective(trial):
            params = {
                "contamination": 0.01,
                "n_estimators":  trial.suggest_int("n_estimators", 5, 100),
                #"max_samples": trial.suggest_float("max_samples", 0.1, 1),
                #"max_features" :trial.suggest_float("max_features", 0.1, 2, log=True)
            }
    
            ILF = IsolationForest(**params)
            ILF.fit(train_data_value, train_data_label)
            ILF_predict = ILF.predict(val_data_value)
        
            for i in range(len(ILF_predict)):
                if ILF_predict[i] == 1:
                    ILF_predict[i] = 0
                else:
                    ILF_predict[i] = 1
    
            return f1_score(val_data_label, ILF_predict)
        
        ## 最佳參數尋找
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)

        ## 計算FineTune後模型各項效能
        def detailed_objective(trial):
            params = {
                "contamination": 0.01,
                "n_estimators":  trial.suggest_int("n_estimators", 5, 100),
                #"max_samples": trial.suggest_float("max_samples", 0.1, 1),
                #"max_features" :trial.suggest_float("max_features", 0.1, 2,log=True)
            }

            ILF = IsolationForest(**params)
            ILF.fit(train_data_value, train_data_label)
            ILF_predict = ILF.predict(val_data_value)
            
            for i in range(len(ILF_predict)):
                if ILF_predict[i] == 1:
                    ILF_predict[i] = 0
                else:
                     ILF_predict[i] = 1

            acc = accuracy_score(val_data_label, ILF_predict)
            recall = recall_score(val_data_label, ILF_predict)
            precision = precision_score(val_data_label, ILF_predict)
            f1 = f1_score(val_data_label, ILF_predict)

            return acc, precision, recall, f1

        ## 印出最佳參數各項效能
        print(f"ILF_group{group}_score: ")
        score_name = ["Accuracy_socre", "Precision_score", "Recall_score", "F1_score"]
        score = ["%.2f" % elem for elem in detailed_objective(study.best_trial)]# 格式化至小數第二位
        score = [float(i) for i in score]# str to float
        print(dict(zip(score_name, score)))
        print()
        #print('Best trial parameters:', study.best_trial.params)


        ## 取最佳參數進行訓練
        best_ILF_model = IsolationForest(
            contamination=0.002,
            n_estimators=study.best_trial.params["n_estimators"],
            #max_samples=study.best_trial.params["max_samples"],
            #max_features=study.best_trial.params["max_features"]
        )                                        
        best_ILF_model.fit(train_data_value)

        ## 儲存模型
        self.__save_model(model=best_ILF_model, model_name=f"ILF_model_group{group}_{self.user_id}")
    
        return detailed_objective(study.best_trial)

    """ 產生ensemble model 訓練集"""
    def __produce_ensemble_data(self, data):
        ensemble_train_data = data.copy()
        
        if (len(ensemble_train_data)>=1000):# 如果超過1000筆就不增加資料
            pass
        else:
            for _ in range(1000//len(ensemble_train_data)):
                ensemble_train_data = pd.concat([ensemble_train_data, data])

        ensemble_train_data = shuffle(ensemble_train_data)# 打亂資料
        ensemble_train_data = ensemble_train_data.reset_index(drop=True)

        return ensemble_train_data

    """ 訓練集乘上random值 """
    def __multiplication_data(self, data):
        for i in range(0,len(data)):
            data.loc[i,::] = data.loc[i,::] * random.uniform(1.01, 1.03)# 將每天資料隨機乘上1.01~1.03

        return data

    """ 壓縮資料 """
    def __compress_data(self, data_L):
        try:
            pca_model = joblib.load(f"./model/{self.user_id}/PCA_model_{self.user_id}.pkl")
        except FileNotFoundError:
            print("找不到PCA模型 ! ")
        data_H = pca_model.transform(data_L)
        
        return data_H

    """ OneClassSVM量化判斷結果 """
    def __OCSVM_AnmoalyRate(self, data, group):
        try:
            OCSVM_model = joblib.load(f"./model/{self.user_id}/OCSVM_model_group{group}_{self.user_id}.pkl")
        except FileNotFoundError:
            print("找不到OneClassSVM模型 ! ")

        score = OCSVM_model.offset_ - OCSVM_model.score_samples(data)
        anomaly_rate = (score/OCSVM_model.offset_)
        anomaly_rate = np.tanh(anomaly_rate)## 大於 0 -> 異常, 小於 0 -> 正常 
    
        return anomaly_rate
        
    """ IsolationForest量化判斷結果 """
    def __ILF_AnmoalyRate(self, data, group):
        try:
            ILF_model = joblib.load(f"./model/{self.user_id}/ILF_model_group{group}_{self.user_id}.pkl")
        except FileNotFoundError:
            print("找不到IsolationForest模型 ! ")
        score = ILF_model.offset_ - ILF_model.score_samples(data)
        anomaly_rate = (score/abs(ILF_model.offset_))
        anomaly_rate = np.tanh(anomaly_rate)## 大於 0 -> 異常, 小於 0 -> 正常 
    
        return anomaly_rate

    """ 兩者判斷結果合併 """
    def __predict_concat(self, OCSVM_predict, ILF_predict):
        ensemble_train_data = np.stack((OCSVM_predict, ILF_predict), axis=1)
        
        return ensemble_train_data
    
    """ ES模型訓練 """
    def __train_es(self, ensemble_train_data, ensemble_train_data_lable, group):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
            patience=50, 
            monitor = 'val_accuracy',
            mode='max')
        ]

        with tf.device('/CPU:0'):# use '/CPU:0' or '/GPU:0'
            ## 搭建模型
            model = Sequential([
                Dense(units=4, input_shape=(2,), activation='relu'),
                Dense(units=1)
            ])
            model.compile(optimizer=Adam(), loss=BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

            history = model.fit(
                ensemble_train_data, 
                ensemble_train_data_lable, 
                validation_split=0.2,
                epochs=100, 
                batch_size=16, 
                shuffle = True,
                callbacks=callbacks,
                verbose=0)

        val = max(history.history['val_accuracy'])# val_data最好分數
        print(f"ES_model_group{group} val_data best score : {val:.2f}")

        ## 儲存模型
        model.save(f'./model/{self.user_id}/ES_model_group{group}_{self.user_id}.h5')

        return


    def fit(self, data, compression_dim=3):
        try:
            dwt_data = self.__wavelet_transform(data)# 小波轉換

            pca_data = self.__pca_preprocess(dwt_data, compression_dim)# PCA特徵縮放

            data_cluster1_H, data_cluster2_H, data_cluster1_L, data_cluster2_L = self.__kmeans_cluster(pca_data, dwt_data)# kemeans分群

            clean_data1_H, outlier_data1_H, clean_data1_L, outlier_data1_L = self.__outlier_detecion(data_cluster1_H, data_cluster1_L)# 雜訊用電事件過濾(cluster1)
            clean_data2_H, outlier_data2_H, clean_data2_L, outlier_data2_L = self.__outlier_detecion(data_cluster2_H, data_cluster2_L)# 雜訊用電事件過濾(cluster2)

            train_data1_H, val_data1_H, train_data1_L, val_data1_L = self.__make_data(clean_data1_H, outlier_data1_H, clean_data1_L, outlier_data1_L)# 製作第一群資料
            train_data2_H, val_data2_H, train_data2_L, val_data2_L = self.__make_data(clean_data2_H, outlier_data2_H, clean_data2_L, outlier_data2_L)# 製作第二群資料

            ## 得出高階特徵資料和標籤
            train_data1_value_H, train_data1_label_H = self.__get_label(train_data1_H)
            val_data1_value_H, val_data1_label_H = self.__get_label(val_data1_H)


            train_data2_value_H, train_data2_label_H = self.__get_label(train_data2_H)
            val_data2_value_H, val_data2_label_H = self.__get_label(val_data2_H)


            ## 訓練Oneclass svm
            self.__train_OCSVM(train_data1_value_H, train_data1_label_H, val_data1_value_H, val_data1_label_H, group=1)
            self.__train_OCSVM(train_data2_value_H, train_data2_label_H, val_data2_value_H, val_data2_label_H, group=2)


            ## 訓練IsolationForest
            self.__train_ILF(train_data1_value_H, train_data1_label_H, val_data1_value_H, val_data1_label_H, group=1)
            self.__train_ILF(train_data2_value_H, train_data2_label_H, val_data2_value_H, val_data2_label_H, group=2)


            ## 製作集成式學習模型資料

            ## 將資料增加至約1000筆
            tmp_data1 = self.__produce_ensemble_data(val_data1_L)# 第一群低階資料
            tmp_data2 = self.__produce_ensemble_data(val_data2_L)# 第二群低階資料

            ## 取出資料與標籤
            ensemble_train_data1_value_L, ensemble_train_data1_lable = self.__get_label(tmp_data1)# ES第一群訓練集資料和標籤
            ensemble_train_data2_value_L, ensemble_train_data2_lable = self.__get_label(tmp_data2)# ES第二群訓練集資料和標籤

            ## 將資料乘上隨機1.01～1.03
            ensemble_train_data1_value_L = self.__multiplication_data(ensemble_train_data1_value_L)
            ensemble_train_data2_value_L = self.__multiplication_data(ensemble_train_data2_value_L)

            ## 壓縮資料
            ensemble_train_data1_value_H = self.__compress_data(ensemble_train_data1_value_L)
            ensemble_train_data2_value_H = self.__compress_data(ensemble_train_data2_value_L)
 
            ## 將訓練資料判斷結果量化-OneClassSVM
            OCSVM_anomaly_rate1 = self.__OCSVM_AnmoalyRate(ensemble_train_data1_value_H, group=1)
            OCSVM_anomaly_rate2 = self.__OCSVM_AnmoalyRate(ensemble_train_data2_value_H, group=2)

            ## 將訓練資料判斷結果量化-IsolationForest4
            ILF_anomaly_rate1 = self.__ILF_AnmoalyRate(ensemble_train_data1_value_H, group=1)
            ILF_anomaly_rate2 = self.__ILF_AnmoalyRate(ensemble_train_data2_value_H, group=2)

            ## 兩者合併
            ensemble_train_data1 = self.__predict_concat(OCSVM_anomaly_rate1, ILF_anomaly_rate1)# 第一群
            ensemble_train_data2 = self.__predict_concat(OCSVM_anomaly_rate2, ILF_anomaly_rate2)# 第二群

            ## ES模型訓練
            self.__train_es(ensemble_train_data1, ensemble_train_data1_lable, group=1)
            self.__train_es(ensemble_train_data2, ensemble_train_data2_lable, group=2)

            print("Train model Success!")
        except:
            print("Train model Fail!")

        return