from helper_functions import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_validate, cross_val_score
from scipy.stats import norm, skew
from scipy import stats

###################################
# Data preparation
###################################

train = pd.read_csv('datasets/housePrice/train.csv')
test = pd.read_csv('datasets/housePrice/test.csv')

train.head()
train.shape # (1460, 81)

test.head()
test.shape # (1459, 80)

df = pd.concat([train, test], ignore_index=True)

###################################
# Exploratory data analysis
###################################

check_df(df)

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

cat_cols
num_cols
cat_but_car

# Analysis of Categorical Variables
for col in cat_cols:
    cat_summary(df, col, plot=True)

# Analysis of Numerical Variables
for col in num_cols:
    num_summary(df, col, plot=True)

# Analysis of Target Variable
for col in cat_cols:
    target_summary_with_cat(df, 'SalePrice', col, plot=True)

# Examination of the dependent variable
df["SalePrice"].hist(bins=100)
plt.show(block=True)

# Examination of the logarithm of the dependent variable
np.log1p(df['SalePrice']).hist(bins=50)
plt.show(block=True)

# Analysis of Correlation
df_corr(df, annot=False)

high_correlated_cols(df, 15)

###################################
# Outliers
###################################

for col in num_cols:
    if col != "SalePrice":
        print(col, ':', check_outlier(df, col))

for col in num_cols:
    if col != "SalePrice":
        boxplot_outliers(df, col)

for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df, col)

###################################
# Missing values and corr
###################################

df.isnull().sum()

missing_values_table(df)

no_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]

for col in no_cols:
    df[col].fillna("No",inplace=True)

df = quick_missing_imp(df, num_method="median", cat_length=17)

missing_values_table(df)

###################################
# Rare analysis and encoding
###################################

rare_analyser(df, "SalePrice", cat_cols)

rare_encoder(df, 0.01)

###################################
# Adding new features
###################################

df.shape # (2919, 81)

df["NEW_1st*GrLiv"] = df["1stFlrSF"] * df["GrLivArea"]

df["NEW_Garage*GrLiv"] = (df["GarageArea"] * df["GrLivArea"])

df["TotalQual"] = df[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                      "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual", "GarageCond", "Fence"]].sum(axis = 1) # 42

# Total Floor
df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]

# Total Finished Basement Area
df["NEW_TotalBsmtFin"] = df["BsmtFinSF1"] + df["BsmtFinSF2"]

# Porch Area
df["NEW_PorchArea"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["ScreenPorch"] + df["3SsnPorch"] + df["WoodDeckSF"]

# Total House Area
df["NEW_TotalHouseArea"] = df["NEW_TotalFlrSF"] + df["TotalBsmtSF"]

df["NEW_TotalSqFeet"] = df["GrLivArea"] + df["TotalBsmtSF"]

# Lot Ratio
df["NEW_LotRatio"] = df["GrLivArea"] / df["LotArea"]

df["NEW_RatioArea"] = df["NEW_TotalHouseArea"] / df["LotArea"]

df["NEW_GarageLotRatio"] = df["GarageArea"] / df["LotArea"]

# MasVnrArea
df["NEW_MasVnrRatio"] = df["MasVnrArea"] / df["NEW_TotalHouseArea"]

# Dif Area
df["NEW_DifArea"] = (df["LotArea"] - df["1stFlrSF"] - df["GarageArea"] - df["NEW_PorchArea"] - df["WoodDeckSF"])

df["NEW_OverallGrade"] = df["OverallQual"] * df["OverallCond"]

df["NEW_Restoration"] = df["YearRemodAdd"] - df["YearBuilt"]

df["NEW_HouseAge"] = df["YrSold"] - df["YearBuilt"]

df["NEW_RestorationAge"] = df["YrSold"] - df["YearRemodAdd"]

df["NEW_GarageAge"] = df["GarageYrBlt"] - df["YearBuilt"]

df["NEW_GarageRestorationAge"] = np.abs(df["GarageYrBlt"] - df["YearRemodAdd"])

df["NEW_GarageSold"] = df["YrSold"] - df["GarageYrBlt"]

df.shape # (2919, 101)

# len(df["Neighborhood"].value_counts())
# df["Neighborhood"].value_counts()
# df.groupby("Neighborhood")["SalePrice"].mean()

###################################
# Label encoding
###################################

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

binary_cols = binary_cols(df)

for col in binary_cols:
    df = label_encoder(df, col)

ohe_cols = [col for col in df.columns if 25 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.shape # (2919, 383)
df.head()

#############################################
# ML model
#############################################

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

y = train_df["SalePrice"]
X = train_df.drop(["Id", "SalePrice"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=17)

models = [('LR', LinearRegression()),
          #('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name})")

# RMSE: 102351485.3432 (LR)
# RMSE: 42778.8298 (CART)
# RMSE: 30264.3698 (RF)
# RMSE: 26197.0809 (GBM)
# RMSE: 28732.3687 (XGBoost)
# RMSE: 29300.6786 (LightGBM)
# RMSE: 25196.2759 (CatBoost)*

#############################################
# ML model with log transformation
#############################################

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

y = np.log1p(train_df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=17)

models = [('LR', LinearRegression()),
          #('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name})")

# RMSE: 511.8367 (LR)
# RMSE: 0.2094 (CART)
# RMSE: 0.1387 (RF)
# RMSE: 0.1276 (GBM)
# RMSE: 0.1414 (XGBoost)
# RMSE: 0.1319 (LightGBM)
# RMSE: 0.1193 (CatBoost)*

catboost_model = CatBoostRegressor(random_state=17).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

new_y_pred = np.expm1(y_pred)
new_y_test = np.expm1(y_test)
np.sqrt(mean_squared_error(new_y_test, new_y_pred))

# RMSE: 25196.2759 (CatBoost) - Before log transformation
# RMSE: 21706.241820062085 (CatBoost) - After log transformation

#############################################
# Hyperparameter tuning * CatBoost
#############################################

catboost_tuning = CatBoostRegressor(random_state=17)
print(catboost_tuning.get_all_params())
rmse1 = np.mean(np.sqrt(-cross_val_score(catboost_tuning, X, y, cv=5, scoring="neg_mean_squared_error")))
# 0.12127948533763588

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_grid_best = GridSearchCV(catboost_tuning, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_grid_best.best_params_ # {'depth': 6, 'iterations': 500, 'learning_rate': 0.1}

final_model = catboost_tuning.set_params(**catboost_grid_best.best_params_).fit(X, y)

rmse2 = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
# 0.12233875778148488

y_final_pred = final_model.predict(X_test)
final_y_pred = np.expm1(y_final_pred)
final_y_test = np.expm1(y_test)
np.sqrt(mean_squared_error(final_y_test, final_y_pred))

# final RMSE = 5942.164020463936

plot_importance(final_model, X)

# Preparing the predictions for submission
predictions = final_model.predict(test_df.drop(["Id","SalePrice"], axis=1))
predictions = np.expm1(predictions)
dictionary = {"Id":test_df["Id"], "SalePrice":predictions}
dfSubmission = pd.DataFrame(dictionary)
dfSubmission.to_csv("housePricePredictions.csv", index=False)

dfSubmission.shape


