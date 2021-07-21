import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from imblearn.over_sampling import SMOTENC
from sklearn.datasets import make_regression




def load_and_preprocess(rel_path):
    path = os.getcwd() + rel_path
    # print(path)
    three_d_df = pd.read_csv(path)
    cat_feats = ['infill_pattern', 'material']
    three_d_df = pd.get_dummies(three_d_df, prefix=cat_feats, drop_first=True)
    three_d_df.rename(columns={'tension_strenght': 'tension_strength'}, inplace=True)

    return three_d_df

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate(grid_predictions, y_test, plot=True):
    # compare predictions to true values
    if plot==True:
        plt.figure(figsize=(6,6))
        x_axis = np.arange(1, len(y_test) + 1)
        plt.scatter(x_axis, grid_predictions, label='pred')
        plt.scatter(x_axis, y_test, label='true')
        plt.ylabel('Tensile Strength (MPa)')
        plt.xlabel('Test Sample (unit)')
        plt.title('Model Predictions')
        plt.legend()
    mape = mean_absolute_percentage_error(y_test, grid_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, grid_predictions))
    r2= r2_score(y_test, grid_predictions)
    print('MAPE: ', mape, '\nRMSE: ', rmse, '\nR2: ', r2)
    return mape, rmse, r2

def generate_data_smote( X, y, num_of_desired_samples = 100):
    # class 2
    X_class2, y_class2 = make_regression(n_samples=num_of_desired_samples, n_features=9, random_state=1)
    y_class2 = np.full((num_of_desired_samples, 1), 2)  # add class assignment

    # class 0,1,2
    X_class012 = np.append(X.to_numpy(), X_class2, axis=0)
    y_class012 = np.append(np.vstack(y.to_numpy()), y_class2, axis=0)

    # check class count
    unique, counts = np.unique(y_class012, return_counts=True)
    # print('Class distribution before smote\n', np.asarray((unique, counts)).T)

    # 2. smote-nc to create new data
    sm = SMOTENC(random_state=2, categorical_features=[7], k_neighbors=2)
    X_res, y_res = sm.fit_resample(X_class012, y_class012)



    # remove class 2
    y_class01 = np.delete(y_res, np.where(y_res == 2), axis=0)
    X_class01 = np.delete(X_res, np.where(y_res == 2), axis=0)
    # return as df
    y_generated = pd.DataFrame(data=y_class01)
    X_generated = pd.DataFrame(data=X_class01)

    unique, counts = np.unique(y_generated, return_counts=True)
    print('Class distribution after smote\n', np.asarray((unique, counts)).T)

    return X_generated, y_generated

def evaluate_data_generation( X_orig, X_gen, plot=True):
    # check difference of correlation matrices of the original and generated dataset
    # a good generation will make this a zero matrix
    corr_matrix = np.add(X_orig.corr(),-X_gen.corr())
    # visualise as heatmap
    # print(corr_matrix.shape)

    if plot==True:
        sns.heatmap(corr_matrix)
        plt.title('Difference of correlation matrices')
        print('Max difference in correlation:\n', np.max(np.abs(corr_matrix).max()))
        print('Average abs difference in correlation:\n', np.mean(np.mean(np.abs(X_gen.corr()-X_orig.corr()))))
    return np.max(np.abs(corr_matrix).max()), np.mean(np.mean(np.abs(X_gen.corr()-X_orig.corr())))


