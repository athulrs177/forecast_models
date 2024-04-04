import pandas as pd
import numpy as np
import xarray as xr
from sklearn.preprocessing import MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import GammaRegressor
from xgboost import XGBRegressor
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from CNN_metric_functions import *

class ForecastModel:
    def __init__(self, params):
        
        self.accum_days = params['accum_days']
        self.wave_id_final = params['wave_id_final']
        
        ###########################################
        # Params for predictor selection
        ###########################################
        self.n_estimators = params['n_estimators']
        self.learning_rateXGB = params['learning_rateXGB']
        self.max_depth = params['max_depth']
        self.objective = params['objective']
        self.min_child_weight = params['min_child_weight']
        self.subsample = params['subsample']
        self.max_features = params['maxFeatures']
        self.threshold = params['threshold']
        self.importance_type = params['importance_type']
        
        self.xgb_regressor = XGBRegressor(n_estimators=self.n_estimators, random_state=42, 
                                          learning_rate=self.learning_rateXGB, max_depth=self.max_depth, 
                                          objective=self.objective, min_child_weight=self.min_child_weight, 
                                          subsample=self.subsample, importance_type=self.importance_type)
        ###########################################
        
        ###########################################
        # Params for Gamma regression model
        ###########################################
        self.max_iter = params['max_iter']
        self.alpha = params['alpha']
        ###########################################
        
        ###########################################
        # Params for CNN model
        ###########################################
        self.hidden_layers = params['hidden_layers']
        self.learning_rate_init = params['learning_rate_init']
        self.loss_CNN = params['loss_CNN']
        self.n_iter_no_change = params['n_iter_no_change']
        self.epochs = params['epochs']
        self.taylor_score_metric = taylor_score_metric
        ###########################################
        
        ###########################################
        # Preprocessors
        ###########################################
        self.scaler = MaxAbsScaler()
        self.imputer = SimpleImputer(keep_empty_features=True)
        ###########################################
    
    def _xgb_fit_predict(self, waves, imerg, train_years):
        
        waves_train = waves[0]
        waves_test = waves[1]
        imerg_train = imerg[0]
        imerg_test = imerg[1]
        
        Selmodel = self.xgb_regressor
        
        selector = SelectFromModel(estimator=Selmodel, threshold=self.threshold, max_features=self.max_features)
        
        ########################################
        # Creating validation data
        ########################################
        eval_frac = 1/len(train_years)
        size = int(imerg_train.shape[0] * (1-eval_frac))
        eval_set=[(waves_train[size:,:], imerg_train[size:])]
        ########################################
        
        ########################################
        # To compute feature importance   
        ########################################
        predictors_train = waves_train.transpose('time', 'waves')
        xgbModel = Selmodel.fit(X=waves_train, y=imerg_train.squeeze(), eval_set = eval_set, verbose=False) 
        feature_importances = Selmodel.feature_importances_
        top_feature_ind = np.argmax(feature_importances)
        top_feature = self.wave_id_final[top_feature_ind]
        ########################################
    
        ########################################
        # Selecting relevant predictors
        ########################################
        waves_train_selected = selector.fit_transform(X=waves_train, y=imerg_train, eval_set=eval_set, verbose=False)
        waves_test_selected = selector.transform(waves_test)
        ########################################
        
        ########################################
        feature_importance_list = {'feature_importance': feature_importances,
                                   'selected_features_mask': selector.get_support(),
                                   'selected_features': self.wave_id_final[selector.get_support()],
                                   'top_features':top_feature,
                                   'n_features_selected': waves_train_selected.shape,
                                   'waves_train': predictors_train
                                  }
        ########################################
        
        return feature_importance_list, waves_train_selected, waves_test_selected
    
    
    def _create_GammaRegression_model(self):
        
        model = GammaRegressor(max_iter=self.max_iter, alpha=self.alpha)
        return model
    
    
    def _create_cnn_model(self, nfeatures):
        tf.random.set_seed(42)  # to replicate results
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', 
                         padding='causal', input_shape=(nfeatures, 1), dilation_rate=1, 
                         kernel_initializer='uniform'))
        model.add(Flatten())
        model.add(Dense(self.hidden_layers[0], activation='relu', kernel_initializer='uniform'))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='relu', kernel_initializer='uniform'))

        optimizer = tf.keras.optimizers.Adamax(learning_rate=self.learning_rate_init)
        model.compile(optimizer=optimizer, loss=self.loss_CNN, metrics=[self.taylor_score_metric])
        
        early_stopping = EarlyStopping(monitor='val_taylor_score_metric', patience=self.n_iter_no_change, 
                                   baseline=None, mode='max', restore_best_weights=True)
        
        return model, early_stopping
    
    def _data_preprocessing(self, waves_train_selected, waves_test_selected, imerg, train_years):
        
        # feature_importance_list, waves_train_selected, waves_test_selected = self._xgb_fit_predict(waves, imerg, train_years)
        
        imerg_train = imerg[0]
        imerg_test = imerg[1]
        
        waves_train_imputed = self.imputer.fit_transform(waves_train_selected)
        waves_test_imputed = self.imputer.transform(waves_test_selected)
        waves_train_scaled = self.scaler.fit_transform(waves_train_imputed)
        waves_test_scaled = self.scaler.transform(waves_test_imputed)
        waves_train_reshaped = waves_train_scaled.reshape(waves_train_scaled.shape[0], waves_train_scaled.shape[1], 1)
        waves_test_reshaped = waves_test_scaled.reshape(waves_test_scaled.shape[0], waves_test_scaled.shape[1], 1)

        imerg_train_imputed = self.imputer.fit_transform(imerg_train.values.reshape(-1,1))
        imerg_test_imputed = self.imputer.transform(imerg_test.values.reshape(-1,1))
        imerg_train_scaled = self.scaler.fit_transform(imerg_train_imputed)
        imerg_test_scaled = self.scaler.transform(imerg_test_imputed)

        return waves_train_reshaped, waves_test_reshaped, imerg_train_scaled, imerg_test_scaled

    
    def GammaRegressor_fit_predict(self, waves, imerg, train_years):
        
        imerg_train = imerg[0]
        imerg_test = imerg[1]
        
        feature_importance_list, waves_train_selected, waves_test_selected = self._xgb_fit_predict(waves, imerg, train_years)
        
        gamma_model = self._create_GammaRegression_model()
        
        history = gamma_model.fit(X=waves_train_selected, y=imerg_train.squeeze())
    
        pred = gamma_model.predict(X=waves_test_selected).squeeze().reshape(imerg_test.shape)        
        
        results = {'predictions': pred,
                   'feature_importance': feature_importance_list['feature_importance'],
                   'selected_features_mask': feature_importance_list['selected_features_mask'],
                   'selected_features': feature_importance_list['selected_features'],
                   'top_features':feature_importance_list['top_features'],
                   'n_features_selected': feature_importance_list['n_features_selected'],
                   'waves_train': feature_importance_list['waves_train'],
                  }
        del gamma_model
        
        return results
        
    def cnn_fit_predict(self, waves, imerg, train_years):
        
        imerg_test = imerg[1]
        
        feature_importance_list, waves_train_selected, waves_test_selected = self._xgb_fit_predict(waves, imerg, train_years)    
            
        waves_train_reshaped, waves_test_reshaped, imerg_train_scaled, imerg_test_scaled = self._data_preprocessing(waves_train_selected, waves_test_selected,
                                                                                                                    imerg, train_years)
        
        model, early_stopping = self._create_cnn_model(nfeatures=int(waves_train_selected.shape[1]))
        
        history = model.fit(x=waves_train_reshaped, y=imerg_train_scaled.squeeze(), 
                            epochs=self.epochs, batch_size=int(92*len(train_years)), 
                            verbose=0, shuffle=False, 
                            validation_split=1/len(train_years), callbacks=[early_stopping])

        pred = model.predict(x=waves_test_reshaped).squeeze().reshape(imerg_test.shape)
        pred = self.scaler.inverse_transform(pred.reshape(-1,1)).squeeze()
        
        results = {'predictions': pred,
                   'feature_importance': feature_importance_list['feature_importance'],
                   'selected_features_mask': feature_importance_list['selected_features_mask'],
                   'selected_features': feature_importance_list['selected_features'],
                   'top_features':feature_importance_list['top_features'],
                   'n_features_selected': feature_importance_list['n_features_selected'],
                   'waves_train': feature_importance_list['waves_train'],
                  }
        del model, early_stopping
        
        return results
