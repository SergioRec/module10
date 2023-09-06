"""Notebook for synthetic control."""
# %%
# imports
import os
import warnings

import pandas as pd

from pyprojroot import here
from module10.synth_en import Placebo, SynthControlCV

# ignore future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# %%
# import processed data
data = pd.read_csv(
    os.path.join(
        here(),
        "data",
        "processed",
        "processed_data.csv",
    ),
    index_col=False,
)

# %%
# filter data as we don't need long time series
data_filt = data[data["date"] >= 201001].copy()

# reduce granularity of data to prevent overfitting
# data_filt = data_filt[data["date"].astype("str").str[-2:].isin(["02", "08"])]

# rank dates to prevent uneven intervals due to date string
data_filt["rank"] = data_filt["date"].rank(method="dense")
rank_treatment = data_filt[data_filt["date"] == 202202]["rank"].unique()[0]

# %%
# keep only low and very low energy dependency units as controls
# keep gas as treated unit
data_filt = data_filt[
    (data_filt["energy_intensity_group"].isin(["Low", "Very low"]))
    | (data_filt["coicop_code"] == "04.5.2")
]


# %%
# params
treated_unit = "GAS "
outcome = "cpi"
time = "rank"
df_id = "item_name"
treatment_date = rank_treatment
features = []

# %%
synth = (
    SynthControlCV(
        data_filt,
        unit=treated_unit,
        outcome=outcome,
        time=time,
        df_id=df_id,
        scale=True,
        treatment_date=treatment_date,
        features=features,
    )
    .prepare_data()
    .reg()
)

synth_df = synth.create_synth()

# %%
print(synth_df)
synth.plot_synth(
    save=True,
    savepath=os.path.join(here(), "outputs"),
    filename="synth_vs.png",
)
synth.plot_dif(
    save=True,
    savepath=os.path.join(here(), "outputs"),
    filename="synth_dif.png",
)

# %%
# placebo tests
placebo = Placebo(
    data_filt,
    treated_unit=treated_unit,
    outcome=outcome,
    time=time,
    df_id=df_id,
    scale=True,
    treatment_date=treatment_date,
    features=features,
)
placebos = placebo.placebo_test()

# %%
placebo.plot_placebo_test(
    filter_bad=True,
    save=True,
    savepath=os.path.join(here(), "outputs"),
    filename="placebos.png",
)

# %%
