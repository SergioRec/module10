"""Class to do synthetic control using ElasticNet."""


import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV  # LassoCV
from sklearn.preprocessing import RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os


class SynthControlCV:
    """Synthetic control object.

    Parameters
    ----------
    df: pandas.DataFrame
        The input dataframe. Needs to be in long format.
    unit: str
        Label for the unit to use as reference.
    outcome: str
        Name of the column containing the outcome variable.
    time: str
        Name of the column containing the time variable.
    df_id: str
        Name of the column containing the labels for each unit.
    treatment_date: int
        Time unit where the treatment started.
    features: list
        List of column names containing variables to include in the model. It
        should not include the outcome.

    Attributes
    ----------
    df
    unit
    outcome
    time
    treatment_date
    features
    id: str
        Name of the column containing the labels for each unit.
    scale: bool
        If true, RobustScaler() will be applied to the data before applying
        the regression. Data will be unscaled before plotting.
    weights_df: pandas.DataFrame
        Dataframe containing the units with a weight different than 0.
    alpha: float
        Optimal alpha value selected for ElasticNet through cross-validation.
    l1_ratio: float
        Optimal ratio between L1 and L2 penalties selected fro ElasticNet
        through cross-validation.
    iterations: int
        Number of iterations needed for ElasticNet to reach convergence.
    mspe: float
        Measure of good fit between synthetic and control during the
        pre-treatment period. Calculated as mean((treated - synth) ** 2).
    df_synth: pandas.DataFrame
        Output dataframe containing the outcome variable for the target unit,
        its synthetic control, the difference between both, and a flag
        that is 1 if there's bad fit pre-treatment.

    Methods
    -------
    prepare_data()
        Scale data and produce X and y arrays for regression.
    reg()
        Apply regression to get weights to create synthetic control.
    create_synth()
        Create output synthetic control dataframe.
    plot_synth(save: bool = False, savepath: str = None, filename: str = None)
        Create lineplot of unit vs synthetic control.
    plot_dif(save: bool = False, savepath: str = None, filename: str = None)
        Create lineplot of the difference between unit and synthetic control,
        including ATT.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        unit: str,
        outcome: str,
        time: str,
        df_id: str,
        treatment_date: int,
        scale: bool = True,
        features: str = [],
    ):
        """Initialize SynthControlCV object."""
        self.df = df.copy()
        self.unit = unit
        self.outcome = outcome
        self.features = [outcome] + features
        self.time = time
        self.id = df_id
        self.treatment_date = treatment_date
        self.scale = scale

    def _scale_data(self):
        """Scale data."""
        self.__scaler = RobustScaler().fit(self.__df_prep[self.features])
        self.__df_prep[self.features] = self.__scaler.transform(
            self.__df_prep[self.features]
        )

    def prepare_data(self):
        """Create X and y arrays from dataframe.

        Returns
        -------
        None

        """
        self.__df_prep = self.df.copy()

        if self.scale is True:
            self._scale_data()

        self.__inverted = (
            # filter pre-intervention period
            self.__df_prep[self.__df_prep[self.time] <= self.treatment_date]
            .pivot(index=self.id, columns=self.time)[
                self.features
            ]  # make one column per year and one row per LA
            .T
        )  # flip the table to have one column per LA

        # treated
        self.__y = self.__inverted[self.unit].values
        # untreated
        self.__X = self.__inverted.drop(columns=self.unit).values

        return self

    def _get_control_weights(self):
        """Create labelled Dataframe to inspect weights assigned to units."""
        array = np.array(self.__inverted.drop(columns=self.unit).columns)
        w_array = self.__weights_array.round(2)
        w_df = pd.DataFrame(
            np.column_stack((array, w_array)), columns=["la", "w"]
        )

        return (
            w_df[abs(w_df["w"]) > 0]
            .reset_index(drop=True)
            .sort_values("w", ascending=False)
        )

    def reg(self):
        """Apply regression to get weights to create synthetic control.

        TODO: Add Lasso option.

        Returns
        -------
        self

        """
        reg = ElasticNetCV(
            fit_intercept=False,
            l1_ratio=list(1.1 - (np.logspace(0, 1, num=100) / 10)),
            max_iter=10000,
            n_jobs=4,
            cv=5,
            tol=1e-2,
        ).fit(self.__X, self.__y)

        self.__weights_array = reg.coef_
        self.weights_df = self._get_control_weights()
        self.alpha = reg.alpha_
        self.l1_ratio = reg.l1_ratio_
        self.iterations = reg.n_iter_

        return self

    def _unscale(self):
        """Unscale data.

        When applying scaler to several columns, it will save a center and
        scale for each one. When applying inverse_transform, it will assume
        that the to-be-unscaled dataframe follows the same structure as the
        original, which is not the case in our analysis as we only have
        outcome variable and synthetic control, which should follow the same
        scaling. So we keep the center and scale that we got for our outcome
        variable only.

        TODO: Add other scalers.
        """

        unscaler = RobustScaler()
        unscaler.center_, unscaler.scale_ = (
            self.__scaler.center_[0],
            self.__scaler.scale_[0],
        )

        self.df_synth[[self.outcome, "synth"]] = unscaler.inverse_transform(
            self.df_synth[[self.outcome, "synth"]]
        )

    def _effect(self):
        """Calculate difference between treated group and synthetic control."""
        self.df_synth["dif"] = (
            self.df_synth[self.outcome] - self.df_synth["synth"]
        )

    def _pred_error(self):
        """Calculates mean squared prediction error (MSPE) pre-treatment."""
        pre = self.df_synth[self.df_synth[self.time] < self.treatment_date]

        self.mspe = np.mean((pre.dif) ** 2)

    def _att(self):
        """Calculates average treatment effect on treated."""
        post = self.df_synth[self.df_synth[self.time] > self.treatment_date]

        self.att = round(np.mean(post["dif"]), 2)

    def create_synth(self):
        """Create synthetic control dataframe.

        Returns
        -------
        pd.DataFrame

        """
        synth_array = (
            self.__df_prep[self.__df_prep[self.id] != self.unit]
            .pivot(index=self.time, columns=self.id)[self.outcome]
            .values.dot(self.__weights_array)
        )

        self.df_synth = self.__df_prep[self.__df_prep[self.id] == self.unit][
            [self.id, self.time, self.outcome]
        ].assign(synth=synth_array)

        if self.scale is True:
            self._unscale()

        self._effect()
        self._pred_error()
        self._att()

        return self.df_synth

    def plot_synth(
        self, save: bool = False, savepath: str = None, filename: str = None
    ):
        """Plot synth vs treated.

        Returns
        -------
        None

        """
        fig, ax = plt.subplots(figsize=(9, 5))
        plt.plot(
            self.df_synth[self.time].unique(),
            self.df_synth[self.outcome].values,
            label=self.unit,
        )
        plt.plot(
            self.df_synth[self.time].unique(),
            self.df_synth["synth"].values,
            label="Synthetic control",
        )
        plt.vlines(
            self.treatment_date,
            ymin=ax.get_ylim()[0],
            ymax=ax.get_ylim()[1],
            linestyles=":",
            colors="black",
            label="Treatment",
        )
        ax.grid(axis="y", color="lightgrey", linewidth=1)
        plt.ylabel(self.outcome)
        plt.xlabel("Time")
        plt.title(self.unit)
        plt.legend(loc="upper left")
        sns.despine()
        if save is True:
            plt.savefig(os.path.join(savepath, filename))
        return ax
        # plt.show()

    def plot_dif(
        self, save: bool = False, savepath: str = None, filename: str = None
    ):
        """Plot effect size.

        Returns
        -------
        None

        """
        fig, ax = plt.subplots(figsize=(9, 5))
        plt.plot(
            self.df_synth[self.time].unique(), self.df_synth["dif"].values
        )

        plt.vlines(
            self.treatment_date,
            ymin=ax.get_ylim()[0],
            ymax=ax.get_ylim()[1],
            linestyles=":",
            colors="black",
            label="Treatment",
        )
        plt.hlines(
            0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors="black"
        )
        ax.grid(axis="y", color="lightgrey", linewidth=1)
        plt.ylabel("Treated - synthetic control")
        plt.xlabel("Time")
        plt.title(self.unit + "\nATT = " + str(self.att))
        plt.legend(loc="upper left")
        sns.despine()
        if save is True:
            plt.savefig(os.path.join(savepath, filename))
        return ax
        # plt.show()


class Placebo(SynthControlCV):
    """TODO: Add docstrings."""

    def __init__(
        self,
        df: pd.DataFrame,
        treated_unit: str,
        outcome: str,
        time: str,
        df_id: str,
        treatment_date: int,
        scale: bool = True,
        features: list = [],
        mspe_adj: int = 10,
    ):
        self.df = df.copy()
        self.outcome = outcome
        self.features = [outcome] + features
        self.time = time
        self.id = df_id
        self.treatment_date = treatment_date
        self.scale = scale
        self.treated = treated_unit
        self.mspe_adj = mspe_adj
        self._dict_outputs = {}  # only for debugging
        self.dict_att = {}
        self.dict_mspe = {}
        self.df_all = pd.DataFrame(
            columns=[
                self.id,
                self.time,
                self.outcome,
                "synth",
                "dif",
                "bad_flag",
            ]
        )

    def _check_pre_fit(self):
        """TODO: docstring."""
        mspe_treated = self.dict_mspe[self.treated]

        for unit in self.dict_mspe:
            if self.dict_mspe[unit] > (mspe_treated * self.mspe_adj):
                self.df_all.loc[self.df_all[self.id] == unit, "bad_flag"] = 1
            else:
                self.df_all.loc[self.df_all[self.id] == unit, "bad_flag"] = 0

    def placebo_test(self):
        """TODO: docstring."""
        for unit in self.df[self.id].unique():
            # print(unit)
            self.unit = unit
            self.prepare_data()
            self.reg()
            self.create_synth()

            self._dict_outputs[unit] = self.__dict__
            self.dict_att[unit] = self.att
            self.dict_mspe[unit] = self.mspe
            self.df_all = pd.concat([self.df_all, self.df_synth])

        self._check_pre_fit()
        return self.df_all

    def _sig(self, att_vals, effect_dir):
        """TODO: docstring."""
        att_treated = self.dict_att[self.treated]

        if effect_dir == "pos":
            self._p = np.round(np.mean(att_vals >= att_treated), 3)
        elif effect_dir == "neg":
            self._p = np.round(np.mean(att_vals <= att_treated), 3)

    def plot_placebo_test(
        self,
        save: bool = False,
        savepath: str = None,
        filename: str = None,
        filter_bad: bool = True,
        effect_dir: str = "pos",
    ):
        """TODO: docstring."""

        if filter_bad is True:
            df = self.df_all[self.df_all["bad_flag"] == 0].copy()
            dict_att_f = {
                key: self.dict_att[key] for key in df[self.id].unique()
            }
            att_vals = np.array(list(dict_att_f.values()))
            self._pool_size = len(df[self.id].unique()) - 1
        else:
            df = self.df_all.copy()
            att_vals = np.array(list(self.dict_att.values()))
            self._pool_size = len(df[self.id].unique()) - 1

        self._sig(att_vals, effect_dir)

        placebos_t = df[df[self.id] == self.treated]
        placebos_p = df[df[self.id] != self.treated]

        p_color = ["grey" for u in placebos_p[self.id].unique()]

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(
            data=placebos_p,
            x=self.time,
            y="dif",
            color="red",
            linewidth=2,
            ax=ax,
            hue=self.id,
            legend=False,
            palette=p_color,
            alpha=0.2,
        )
        sns.lineplot(
            data=placebos_t,
            x=self.time,
            y="dif",
            color="red",
            linewidth=2,
            ax=ax,
            legend=False,
        )

        plt.vlines(
            self.treatment_date,
            ymin=ax.get_ylim()[0],
            ymax=ax.get_ylim()[1],
            linestyles=":",
            colors="black",
            label="Treatment",
        )
        plt.hlines(
            y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], color="black"
        )
        plt.title(
            self.treated
            + "\np = "
            + str(self._p)
            + "\npool size = "
            + str(self._pool_size)
        )
        ax.grid(axis="y", color="lightgrey", linewidth=1)
        sns.despine()
        if save is True:
            plt.savefig(os.path.join(savepath, filename))
        return ax
        # plt.show()
