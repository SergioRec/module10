"""Notebook for synthetic control."""
# %%
# imports
import os
import warnings

import matplotlib.pyplot as plt
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

########################
#         GAS          #
########################
# %%
# keep only low and very low energy dependency units as controls
# keep gas as treated unit
data_gas = data_filt[
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
        data_gas,
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
print(synth_df)
# %%
ax = synth.plot_synth(
    save=True,
    savepath=os.path.join(here(), "outputs"),
    filename="gas_synth_vs.png",
)
ax.set_xticks(list(data_gas["rank"].unique())[1::16])
ax.set_xticklabels([str(d) for d in list(data_gas["date"].unique())[1::16]])
plt.show()


# %%
ax = synth.plot_dif(
    save=True,
    savepath=os.path.join(here(), "outputs"),
    filename="gas_synth_dif.png",
)
ax.set_xticks(list(data_gas["rank"].unique())[1::16])
ax.set_xticklabels([str(d) for d in list(data_gas["date"].unique())[1::16]])
plt.show()

# %%
# placebo tests
placebo = Placebo(
    data_gas,
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
ax = placebo.plot_placebo_test(
    filter_bad=True,
    save=True,
    savepath=os.path.join(here(), "outputs"),
    filename="gas_placebos.png",
)
ax.set_xticks(list(data_gas["rank"].unique())[1::16])
ax.set_xticklabels([str(d) for d in list(data_gas["date"].unique())[1::16]])
plt.show()

# %%
########################
#  FUELS & LUBRICANTS  #
########################
# %%
# keep only high and very high energy dependency units as controls
# keep gas as treated unit
data_fuel = data_filt[
    (data_filt["energy_intensity_group"].isin(["High", "Very high"]))
    | (data_filt["coicop_code"] == "01.1.5")
]


# %%
# params
treated_unit = "OILS & FATS "
outcome = "cpi"
time = "rank"
df_id = "item_name"
treatment_date = rank_treatment
features = ["energy_intensity"]

# %%
synth = (
    SynthControlCV(
        data_fuel,
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
print(synth_df)

# %%
ax = synth.plot_synth(
    save=True,
    savepath=os.path.join(here(), "outputs"),
    filename="cook_oil_synth_vs.png",
)
ax.set_xticks(list(data_fuel["rank"].unique())[1::16])
ax.set_xticklabels([str(d) for d in list(data_fuel["date"].unique())[1::16]])
plt.show()

# %%
ax = synth.plot_dif(
    save=True,
    savepath=os.path.join(here(), "outputs"),
    filename="cook_oil_synth_dif.png",
)
ax.set_xticks(list(data_fuel["rank"].unique())[1::16])
ax.set_xticklabels([str(d) for d in list(data_fuel["date"].unique())[1::16]])
plt.show()

# %%
# placebo tests
placebo = Placebo(
    data_fuel,
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
ax = placebo.plot_placebo_test(
    filter_bad=True,
    save=True,
    savepath=os.path.join(here(), "outputs"),
    filename="cook_oil_placebos.png",
)
ax.set_xticks(list(data_fuel["rank"].unique())[1::16])
ax.set_xticklabels([str(d) for d in list(data_fuel["date"].unique())[1::16]])
plt.show()
# %%
