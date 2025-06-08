"""Playground/cheat sheet for local data analytics and visualization."""

import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import (
    train_test_split as sk_train_test_split,
    cross_val_score,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from .fetch_data import load_dataset
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer

import pandas as pd


def get_stats(ds=None):
    """Show common dataset values to get some insight on the data."""

    dataset = ds or load_dataset()
    print("get_stats")
    print("dataset", dataset)
    print("dataset.head", dataset.head())
    print("dataset.info", dataset.info())
    print("dataset.describe", dataset.describe())
    print("non-number fields:")
    print("Weather", dataset['Weather'].value_counts())
    print("Traffic_Conditions", dataset['Traffic_Conditions'].value_counts())
    print("Day_of_Week", dataset['Day_of_Week'].value_counts())
    print("Time_of_Day", dataset['Time_of_Day'].value_counts())
    corr_matrix = dataset.corr(numeric_only=True)
    print(
        "correlation matrix:",
        corr_matrix["Trip_Price"].sort_values(
            ascending=False
        ),
    )


def build_plots(ds=None):
    """
    Examples of plots to evaluate data visually, e.g. to look for correlations.
    """

    dataset = ds or load_dataset()
    # print("build_plots")
    dataset.hist(bins=50, figsize=(12, 8))
    plt.show()
    # dataset["distance_cat"].value_counts().sort_index().plot.bar(
    # rot=0, grid=True)
    # plt.xlabel("Distance category")
    # plt.ylabel("Number of trips")
    # plt.show()
    # attributes = [
    #     "Trip_Price",
    #     "Trip_Distance_km",
    #     "Per_Km_Rate",
    #     "Trip_Duration_Minutes",
    #     "Per_Minute_Rate",
    #     "Base_Fare",
    #     "Passenger_Count",
    #     # todo: test enums
    #     # "Time_of_Day",
    #     # ""
    # ]
    # scatter_matrix(dataset[attributes], figsize=(12, 8))
    # plt.show()


# def get_data_preprocessor():
#     num_pipeline = make_pipeline(
#         SimpleImputer(strategy="median"),
#         StandardScaler()
#     )
#     cat_pipeline = make_pipeline(
#         SimpleImputer(strategy="most_frequent"),
#         OneHotEncoder(handle_unknown="ignore")
#     )
#     preprocessing = ColumnTransformer(
#         [
#             (
#               "cat",
#               cat_pipeline,
#               make_column_selector(dtype_include=object)
#             ),
#         ],
#         remainder=num_pipeline
#     )
#     return preprocessing


def preprocess_data(ds=None):
    """Prepare the entire dataset to be more ML-friendly."""

    dataset = ds if ds is not None else load_dataset()
    numeric_cols = dataset.select_dtypes(include='number').columns
    object_cols = dataset.select_dtypes(include='object').columns

    # fill all numeric values with median values
    num_imputer = SimpleImputer(strategy='median')
    dataset[numeric_cols] = num_imputer.fit_transform(dataset[numeric_cols])

    # fill all "enum" values with most frequent values
    obj_imputer = SimpleImputer(strategy='most_frequent')
    dataset[object_cols] = obj_imputer.fit_transform(dataset[object_cols])

    # turn "enum" values into a set of 1/0 fields
    dataset = pd.get_dummies(
        dataset,
        columns=[
            'Time_of_Day',
            'Day_of_Week',
            'Traffic_Conditions',
            'Weather'
        ],
        drop_first=True
    )

    # slice trip prices off the dataset.
    # They will be our reference labels.
    dataset_labels = dataset["Trip_Price"].copy()
    dataset = dataset.drop("Trip_Price", axis=1)

    # scale values to be more ML-friendly
    scaler = RobustScaler()
    dataset_scaled = dataset.copy()

    # refresh numeric cols after we excluded Trip_Price
    numeric_cols = dataset.select_dtypes(include='number').columns

    dataset_scaled[numeric_cols] = scaler.fit_transform(dataset[numeric_cols])

    return {
        "dataset": dataset,
        "dataset_labels": dataset_labels,
    }


def train_test_split(dataset, labels, test_size=0.2):
    dataset_train, dataset_test, label_train, label_test = sk_train_test_split(
        dataset,
        labels,
        test_size=test_size,
    )
    return {
        "dataset_train": dataset_train,
        "dataset_test": dataset_test,
        "label_train": label_train,
        "label_test": label_test
    }


def train_model(ds=None):
    """
    A collection of different ML models that was tested.

    The most efficient of them is left uncommented, to be used in real app.
    """

    dataset = ds or load_dataset()
    preprocess_res = preprocess_data(dataset)
    split_res = train_test_split(
        preprocess_res["dataset"],
        preprocess_res["dataset_labels"]
    )
    # data = preprocess_data(ds)

    # preprocessing = get_data_preprocessor()
    # lin_reg = LinearRegression()
    # lin_reg.fit(split_res["dataset_train"], split_res["label_train"])
    # dataset_predictions = lin_reg.predict(split_res["dataset_train"])

    # tree_reg = DecisionTreeRegressor(random_state=42)
    # tree_reg.fit(split_res["dataset_train"], split_res["label_train"])
    # dataset_predictions = tree_reg.predict(split_res["dataset_train"])
    # tree_rmses = -cross_val_score(
    #   tree_reg,
    #   split_res["dataset_train"],
    #   split_res["label_train"],
    #   scoring="neg_root_mean_squared_error",
    #   cv=10
    # )
    # print("tree_rmses:", tree_rmses)

    # print("predictions", dataset_predictions[:5])
    # print("labels", split_res["label_train"].iloc[:5].values)
    # err = mean_absolute_error(
    #     split_res["label_train"],
    #     dataset_predictions,
    # )
    # print("err:", err)

    forest_reg = RandomForestRegressor(random_state=42, max_features=10)
    forest_rmses = -cross_val_score(
        forest_reg,
        split_res["dataset_train"],
        split_res["label_train"],
        scoring="neg_root_mean_squared_error",
        cv=10
    )
    print("forest_rmses:", forest_rmses)
    # import pdb; pdb.set_trace()
    forest_reg.fit(split_res["dataset_train"], split_res["label_train"])
    dataset_predictions = forest_reg.predict(split_res["dataset_train"])
    tree_rmses = -cross_val_score(
        forest_reg,
        split_res["dataset_train"],
        split_res["label_train"],
        scoring="neg_root_mean_squared_error",
        cv=10
    )
    print("tree_rmses:", tree_rmses)
    err = mean_absolute_error(
        split_res["label_train"],
        dataset_predictions,
    )
    print("err:", err)

    final_predictions = forest_reg.predict(split_res["dataset_test"])
    final_error = mean_absolute_error(
        split_res["label_test"], final_predictions)
    print("final_error", final_error)

    return forest_reg


def dump_model():
    model = train_model()
    joblib.dump(model, "model.pkl")


def get_model():
    model = joblib.load("model.pkl")
    # dataset = load_dataset()
    # preprocess_res = preprocess_data(dataset)
    # split_res = train_test_split(
    #     preprocess_res["dataset"],
    #     preprocess_res["dataset_labels"]
    # )
    # predictions = model.predict(split_res["dataset_test"])
    # error = mean_absolute_error(
    #     split_res["label_test"], predictions)
    # print("error", error)
    return model
