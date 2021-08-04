from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import (
    Pipeline,
    FeatureUnion,
    _fit_transform_one,
    _transform_one,
    Parallel,
    delayed,
    sparse,
)
import pandas as pd
import numpy as np

feat_type = {
    "categorical": ["year", "modelyr", "crashtm"],
    "ordinal": [
        "passgcar",
        "suv",
        "weekend",
        "frimp",
        "indfrimp",
        "rearimp",
        "indrearimp",
        "rsimp",
        "lsimp",
        "impchldst",
        "impstblt",
        "ruralrd",
        "row1",
        "backright",
        "backleft",
        "backother",
        "male",
        "vehicle1",
        "vehicle2",
        "missweight",
        "drivebelt",
        "splmU55",
        "lowviol",
        "highviol",
        "car_age"
    ],
    "numeric": ["thoulbs_I", "numcrash"],
}


def get_feats(df, feature_list):
    """ """
    return [col for col in df.columns if col in feature_list]


class NanFiller(BaseEstimator, TransformerMixin):
    """
    Transformer: Fills missing values with 0, and adds a dummy for each variable
    for which the share of missing observations is higher than
    `min_nan_proportion` indicating whether the variable was missing in the
    original data
    """

    def __init__(self, min_nan_proportion=0.01) -> None:
        super().__init__()
        self.feats_to_nandummy = None
        self.min_nan_proportion = min_nan_proportion

    def fit(self, X, y=None):
        self.feats_to_nandummy = [
            feat
            for feat, val in (X.isna().mean() > self.min_nan_proportion).items()
            if val
        ]
        return self

    def transform(self, X, y=None):
        Xt = X.copy(deep=True)
        featnanames = [feat + "_nan" for feat in self.feats_to_nandummy]
        Xt[featnanames] = Xt[self.feats_to_nandummy].isna().astype(int)
        Xt = Xt.fillna(0)
        return Xt


class RareAggregator(BaseEstimator, TransformerMixin):
    """ """

    def __init__(self, tolerance=0.01) -> None:
        super().__init__()
        self.recode = {}
        self.tolerance = tolerance

    def fit(self, X, y=None):
        for feat in X.columns:
            to_aggr = X[feat].value_counts() / X.shape[0] <= 0.01
            if to_aggr.sum() > 0:
                self.recode[feat] = {
                    val: "other" if aggr else str(val)
                    for val, aggr in dict(to_aggr).items()
                }
        return self

    def transform(self, X, y=None):
        Xt = X.copy(deep=True)
        Xt = Xt.apply(
            lambda x: x.map(self.recode[x.name]) if x.name in self.recode.keys() else x
        )
        return Xt


class OneHotPd(OneHotEncoder):
    """ """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, X):
        return pd.DataFrame(
            super().transform(X), columns=super().get_feature_names(X.columns)
        )


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects given features from a pandas DataFrame
    """

    def __init__(self, input_feats):
        self.input_feats = input_feats
        self.selected_features = None

    def fit(self, X, y=None):
        if type(X) is pd.core.frame.DataFrame:
            self.selected_features = [
                feat for feat in self.input_feats if feat in X.columns
            ]
        return self

    def transform(self, X, y=None):
        if type(X) is pd.core.frame.DataFrame:
            return X.loc[:, self.selected_features]
        else:
            print(
                "Warning: Input features are not a DataFrame. FeatureSelector is not doing anything"
            )
            return X


class PandasFeatureUnion(FeatureUnion):
    """
    Clone of sklearn's FeatureUnion, but it returns a pd.DataFrame. Only to be
    used if previous transformers also return pd.DataFrames
    """

    def fit_transform(self, X, y=None, **fit_params):
        results = self._parallel_func(X, y, fit_params, _fit_transform_one)
        if not results:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        cols = []
        for x in Xs:
            x.index = range(x.shape[0])
            cols += list(x.columns)
        X = pd.concat(Xs, axis="columns", copy=False, ignore_index=True)
        X.columns = cols
        return X

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter()
        )
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs


def construct_data_pipeline(pd_output=False, feat_type=feat_type):
    """
    Returns a default sklearn pipeline for pre-processing data
    """
    categorical_pipeline = Pipeline(
        steps=[
            ("cat_selector", FeatureSelector(feat_type["categorical"])),
            ("rare_cat", RareAggregator()),
            ("one_hot", OneHotPd(handle_unknown="ignore", sparse=False)),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("cat_selector", FeatureSelector(feat_type["numeric"])),
            ("nan_filler", NanFiller()),
        ]
    )
    ordinal_pipeline = Pipeline(
        steps=[
            ("cat_selector", FeatureSelector(feat_type["ordinal"])),
            ("nan_filler", NanFiller()),
        ]
    )
    tlist = [
        ("categorical_pipeline", categorical_pipeline),
        ("numeric_pipeline", numeric_pipeline),
        ("ordinal_pipeline", ordinal_pipeline),
    ]
    if pd_output:
        data_pipeline = PandasFeatureUnion(transformer_list=tlist)
    else:
        data_pipeline = FeatureUnion(transformer_list=tlist)
    return data_pipeline
