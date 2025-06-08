import kagglehub
from kagglehub import KaggleDatasetAdapter


def load_dataset():
    # Download latest version
    path = kagglehub.dataset_download("denkuznetz/taxi-price-prediction")
    dataset = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "denkuznetz/taxi-price-prediction",
        "taxi_trip_pricing.csv"
    )

    print("Path to dataset files:", path)

    # there are only 70 rows above 50km, and all of them are actually 100+ km.
    # Those are outliers and they will mess up our model, so we remove them.
    dataset = dataset[dataset["Trip_Distance_km"] < 51]

    return dataset
