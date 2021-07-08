import pandas as pd
import numpy as np


class DML_diagnostics:
    """
    Makes it possible to construct diagnostics from an `econml.DML` model. For
    example, `econml.DML` does not store first stage R2s. To estimate them, one
    needs to reconstruct the models pairing with the help of a cv_splitter
    """

    def __init__(self, cv_splitter):
        self.cv_splitter = cv_splitter

    @staticmethod
    def _select(data, idx):
        """
        Flexibly selects a subset of data given a positional index, whether
        `data` is a 2-dimensional numpy array or a pd.DataFrame
        """
        if type(data) in (pd.core.frame.DataFrame, pd.core.series.Series):
            return data.iloc[idx]
        else:
            return data[idx]

    def get_ate(model):
        """
        Returns ATE of model and confidence interval. Model needs to be a DML model
        instance
        """
        return model.ate(), model.ate_interval()

    def get_score_T(self, model, scoring_function, W, T):
        """
        Computes the first stage `scoring_function` from a DML model for the
        treatment variable, where `scoring_function` is a sklearn scoring
        function. Assumes that the `cv_splitter` used in the training of the
        model has been the same used when instantiating the `DML_diagnostics`
        class
        """
        scores = []
        for k, (train, test) in enumerate(self.cv_splitter.split(W)):
            if model.discrete_treatment:
                pred = model.models_t[0][k].predict_proba(self._select(W, test))[:, 1]
            else:
                pred = model.models_t[0][k].predict(self._select(W, test))
            scores += [scoring_function(self._select(T, test), pred)]
        return np.mean(scores)

    def get_score_y(self, model, scoring_function, W, T):
        """
        Computes the first stage `scoring_function` from a DML model for the
        treatment variable, where `scoring_function` is a sklearn scoring
        function. Assumes that the `cv_splitter` used in the training of the
        model has been the same used when instantiating the `DML_diagnostics`
        class
        """
        scores = []
        for k, (train, test) in enumerate(self.cv_splitter.split(W)):
            pred = model.models_y[0][k].predict(self._select(W, test))
            scores += [scoring_function(self._select(T, test), pred)]
        return np.mean(scores)

    def first_stage_residuals(self, model, y, T, W=None, X=None):
        """
        Computes the first stage residuals from a DML model. Assumes that the
        `cv_splitter` used in the training of the model has been the same used
        when instantiating the `DML_diagnostics` class
        """
        if W is not None:
            if X is not None:
                controls = np.hstack([X, W])
            else:
                controls = W
        else:
            if X is not None:
                controls = X
            else:
                controls = None

        res = []
        for k, (train, test) in enumerate(self.cv_splitter.split(controls)):
            pred_func = model.models_t[0][k].predict_proba if model.discrete_treatment else model.models_t[0][k].predict
            res.append(
                pd.DataFrame(
                    np.vstack(
                        [
                            self._select(y, test) - model.models_y[0][k].predict(self._select(controls, test)),
                            self._select(T, test) - pred_func(self._select(controls, test)).reshape(-1, 1)[:, -1],
                        ]
                    ).T,
                    index=test,
                    columns=["y", "T"],
                )
            )

        return pd.concat(res, axis="index").sort_index()

    def reorder_df_to_splitter(self, df):
        """
        Reorder a df according to the cv splitter used in the DML model, to
        align original data rows to residuals and predictions.
        """
        return pd.concat(
            [pd.DataFrame(self._select(df, test)) for _, test in self.cv_splitter.split(df)],
            axis=0
        ).sort_index()
