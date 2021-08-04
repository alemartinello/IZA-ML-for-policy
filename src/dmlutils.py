import pandas as pd
import numpy as np


class DML_diagnostics:
    """
    Makes it possible to construct diagnostics from an `econml.DML` model. For
    example, `econml.DML` does not store first stage R2s. To estimate them, one
    needs to reconstruct the models pairing with the help of a cv_splitter
    """

    def __init__(self, model):
        self.model = model

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

    def predict_T(self, W=None, X=None, predict_proba="auto"):
        """ """
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
        if predict_proba == "auto":
            predict_proba = self.model.discrete_treatment
        else:
            predict_proba = predict_proba

        res = []
        for k, (train, test) in enumerate(self.model.cv.split(controls)):
            pred_func = (
                self.model.models_t[0][k].predict_proba
                if predict_proba
                else self.model.models_t[0][k].predict
            )
            if predict_proba:
                res.append(
                    pd.DataFrame(
                        1 - pred_func(self._select(controls, test))[:, 0].reshape(-1, 1),
                        columns=["T_pred"],
                        index=test,
                    )
                )
            else:
                res.append(
                    pd.DataFrame(
                        pred_func(self._select(controls, test)),
                        columns=["T_pred"],
                        index=test,
                    )
                )

        return pd.concat(res, axis=0).sort_index()

    def predict_y(self, W=None, X=None, predict_proba="auto"):
        """ """
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
        if predict_proba == "auto":
            predict_proba = False

        res = []
        for k, (train, test) in enumerate(self.model.cv.split(controls)):
            pred_func = (
                self.model.models_y[0][k].predict_proba
                if predict_proba
                else self.model.models_y[0][k].predict
            )
            if predict_proba:
                res.append(
                    pd.DataFrame(
                        pred_func(self._select(controls, test))[:, -1].reshape(-1, 1),
                        columns=["y_pred"],
                        index=test,
                    )
                )
            else:
                res.append(
                    pd.DataFrame(
                        pred_func(self._select(controls, test)),
                        columns=["y_pred"],
                        index=test,
                    )
                )

        return pd.concat(res, axis=0).sort_index()

    def first_stage_residuals(self, y, T, W=None, X=None):
        """
        Computes the first stage residuals from a DML model. Assumes that the
        `cv_splitter` used in the training of the model has been the same used
        when instantiating the `DML_diagnostics` class
        """
        res = pd.DataFrame(np.vstack([
            (T.values - self.predict_T(X=X)['T_pred'].values),
            (y.values - self.predict_y(X=X)['y_pred'].values)
        ]).T, columns=['T', 'y'])

        return res

    def reorder_df_to_splitter(self, df):
        """
        Reorder a df according to the cv splitter used in the DML model, to
        align original data rows to residuals and predictions.
        """
        return pd.concat(
            [
                pd.DataFrame(self._select(df, test))
                for _, test in self.model.cv.split(df)
            ],
            axis=0,
        ).sort_index()


def binscatter(x, y, by=None, nbins=50):
    """
    """
    toplot = pd.concat([y, x], axis=1, ignore_index=True)
    toplot.columns = ['y', 'x']
    if by is None:
        output = toplot.groupby(pd.qcut(toplot['x'], 50, duplicates='drop', labels=False)).agg({'y': 'mean', 'x': ['mean', 'count']})
        output.columns = ['y', 'x', 'count']
    else:
        dflist = []
        for label in set(by):
            selection = (by == label)
            selection = np.array(selection)
            t = toplot[selection]
            t = t.groupby(pd.qcut(t['x'], 50, duplicates='drop', labels=False)).agg({'y': 'mean', 'x': ['mean', 'count']})
            t.columns = ['y', 'x', 'count']
            t['by'] = label
            dflist.append(t)
        output = pd.concat(dflist, axis=0)
    return output
