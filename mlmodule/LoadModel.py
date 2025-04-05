import os
import numpy as np
import pandas as pd
import joblib
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'# 關閉tensorflow訊息
import tensorflow as tf
from mlmodule.preprocess_method import DWT_denoise
from keras.models import load_model


class Model_load():
    def __init__(self, user_id):
        self.user_id = user_id

    """ 資料處裡 """
    def __data_preprocess(self, data):
        
        trans_data = pd.DataFrame(data)
        trans_data = trans_data/1000# 瓦轉千瓦
        trans_data = np.round(trans_data, 2)# 四捨五入至第二位

        return trans_data

    """ 載入模型 """
    def __load_model(self, model_name):
        load_path = f"mlmodule/model/{self.user_id}/{model_name}.pkl"
        try:
            model = joblib.load(load_path)
        except:
            print(f"找不到{self.user_id,model_name,load_path}")

        return model

    """ 小波轉換 """
    def __wavelet_transform(self, data):
        tmp_data = copy.deepcopy(data)
        de = DWT_denoise()
        ## 小波轉換
        date = data.iloc[:,0]# 保留原有日期
        date= pd.DataFrame(date)
        tmp_data.drop(["Report_time"], axis=1, inplace=True)# 刪除日期
        dwt_data = de.transform(tmp_data.values, 'db4', level=1)
        dwt_data = self.__data_preprocess(dwt_data)


        return  dwt_data, date

    """ PCA特徵縮放 """
    def __pca_preprocess(self, data):
        
        ## PCA轉換
        pca_model = self.__load_model(f'PCA_model_{self.user_id}')
        ## 載入已訓練好模型
        pca_data = pca_model.transform(data.values)
        pca_data = pd.DataFrame(data=pca_data)

        return pca_data    

    """ 提取分群後資料 """
    def __get_ClusterData(self, data, data_label):
        tmp_data = copy.deepcopy(data)
        tmp_data["cluster_label"] = data_label
        
        # 獲取第一群資料   
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

    """ 載入kmeans model 分群 """
    def __kmeans_cluster(self, data, date):
        ## 載入kmeans model
        kmeans_model = self.__load_model(f'kmeans_model_{self.user_id}')

        kmeans_label = kmeans_model.predict(data)

        data_cluster1, data_cluster2 = self.__get_ClusterData(data, kmeans_label)# 取出各群資料
        date_cluster1, date_cluster2 = self.__get_ClusterData(date, kmeans_label)# 取出各群資料

        return data_cluster1, data_cluster2, date_cluster1, date_cluster2

    """ OneClassSVM量化判斷結果 """
    def __OCSVM_AnmoalyRate(self, data, group):
        try:
            OCSVM_model = joblib.load(f"mlmodule/model/{self.user_id}/OCSVM_model_group{group}_{self.user_id}.pkl")
        except FileNotFoundError:
            print("找不到OneClassSVM模型 ! ")

        score = OCSVM_model.offset_ - OCSVM_model.score_samples(data)
        anomaly_rate = (score/OCSVM_model.offset_)
        anomaly_rate = np.tanh(anomaly_rate)## 大於 0 -> 異常, 小於 0 -> 正常 
    
        return anomaly_rate

    """ IsolationForest量化判斷結果 """
    def __ILF_AnmoalyRate(self, data, group):
        try:
            ILF_model = joblib.load(f"mlmodule/model/{self.user_id}/ILF_model_group{group}_{self.user_id}.pkl")
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
    
    """ 結果轉換 """
    def __trans_predict(self, predict):
        ## 大於0->1(異常), 小於0->0(正常)
        for i in range(len(predict)):
            if predict[i]>0:
                predict[i] = 1
            else:
                predict[i] = 0

        return predict


    """ ES檢測 """
    def __ES_predict(self, annomaly_rate, group):
        try:
            ES_model = load_model(f'mlmodule/model/{self.user_id}/ES_model_group{group}_{self.user_id}.h5')
        except FileNotFoundError:
            print("找不到ES模型 ! ")

        predict = ES_model.predict(annomaly_rate)
        predict = self.__trans_predict(predict)
        predict = pd.DataFrame(predict, columns=["label"])

        return predict

    """ 裝上日期 """
    def __date_preprorcess(self, predict, date):

        predict = pd.DataFrame(predict)
        predict["Report_time"] = date

        return predict


    """ 集合兩者檢測結果 """
    def __predict_preprocess(self, predict1, predict2, date1, date2):

        ## 裝上日期
        predict1 = self.__date_preprorcess(predict1, date1)
        predict2 = self.__date_preprorcess(predict2, date2)

        predict = pd.concat([predict1, predict2],axis=0)# 兩者結合
        predict = predict.sort_values(by="Report_time")# 按日期重新排序
        predict = predict.drop(["Report_time"], axis=1)# 刪除日期

        return predict


    def predict(self, data):
        dwt_data, date = self.__wavelet_transform(data)# 小波轉換

        pca_data = self.__pca_preprocess(dwt_data)# PCA特徵縮放

        data_cluster1, data_cluster2, date_cluster1, date_cluster2 = self.__kmeans_cluster(pca_data, date)# kemeans分群

        first = data_cluster1.empty or data_cluster2.empty
        if(first):
            if (not(data_cluster1.empty)):
                ## Oneclass SVM檢測
                OCSVM_anomaly_rate1 = self.__OCSVM_AnmoalyRate(data_cluster1, group=1)

                ## IsolatinForest檢測
                ILF_anomaly_rate1 = self.__ILF_AnmoalyRate(data_cluster1, group=1)

                ## 單類分類器檢測結果合併
                ensemble_data1 = self.__predict_concat(OCSVM_anomaly_rate1, ILF_anomaly_rate1)# 第一群

                ## ES檢測
                group1_predict = self.__ES_predict(ensemble_data1, group=1)

                ## 結合兩者檢測結果
                #predict = self.__predict_preprocess(group1_predict, group2_predict, date_cluster1, date_cluster2)

                return group1_predict

            elif (not(data_cluster2.empty)):
                ## Oneclass SVM檢測
                OCSVM_anomaly_rate2 = self.__OCSVM_AnmoalyRate(data_cluster2, group=2)

                ## IsolatinForest檢測
                ILF_anomaly_rate2 = self.__ILF_AnmoalyRate(data_cluster2, group=2)

                ## 單類分類器檢測結果合併
                ensemble_data2 = self.__predict_concat(OCSVM_anomaly_rate2, ILF_anomaly_rate2)# 第一群

                ## ES檢測
                group2_predict = self.__ES_predict(ensemble_data2, group=2)

                ## 結合兩者檢測結果
                #predict = self.__predict_preprocess(group1_predict, group2_predict, date_cluster1, date_cluster2)

                return group2_predict

        else:
            ## Oneclass SVM檢測
            OCSVM_anomaly_rate1 = self.__OCSVM_AnmoalyRate(data_cluster1, group=1)
            OCSVM_anomaly_rate2 = self.__OCSVM_AnmoalyRate(data_cluster2, group=2)

            ## IsolatinForest檢測
            ILF_anomaly_rate1 = self.__ILF_AnmoalyRate(data_cluster1, group=1)
            ILF_anomaly_rate2 = self.__ILF_AnmoalyRate(data_cluster2, group=2)

            ## 單類分類器檢測結果合併
            ensemble_data1 = self.__predict_concat(OCSVM_anomaly_rate1, ILF_anomaly_rate1)# 第一群
            ensemble_data2 = self.__predict_concat(OCSVM_anomaly_rate2, ILF_anomaly_rate2)# 第二群

            ## ES檢測
            group1_predict = self.__ES_predict(ensemble_data1, group=1)
            group2_predict = self.__ES_predict(ensemble_data2, group=2)

            ## 結合兩者檢測結果
            predict = self.__predict_preprocess(group1_predict, group2_predict, date_cluster1, date_cluster2)

            return predict
