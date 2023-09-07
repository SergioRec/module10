"""Notebook for data pre-processing."""

# %%
# imports
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pyprojroot import here

# ignore future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
# %%
# load data files

## CPI tables: https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/consumerpriceinflation  # noqa
cpi = pd.read_excel(
    os.path.join(
        here(),
        "data",
        "external",
        "consumerpriceinflationdetailedreferencetables.xlsx",
    ),
    skiprows=[0, 1, 2, 3, 5, 6],
    skipfooter=11,
    usecols="A:DT",
    sheet_name="Table 57",
    na_values="..",
)

## Energy intensity: https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/contributionstotheconsumerpricesindexcpibyenergyintensity  # noqa
ei = pd.read_excel(
    os.path.join(here(), "data", "external", "cddataset.xlsx"), skiprows=2
)

# %%
# process horrible excel format
ei[["coicop_code", "item_name"]] = ei["COICOP Class level item"].str.split(
    " : ", expand=True
)
ei = ei.drop("COICOP Class level item", axis=1)
ei = ei.rename(
    columns={
        "Energy Intensity": "energy_intensity",
        "Energy Intensity Group": "energy_intensity_group",
    }
)

# %%
cpi = cpi.rename(columns={"Unnamed: 0": "date"}).drop(
    columns="aggregate number"
)
cpi_melted = cpi.melt(id_vars="date", var_name="coicop_code", value_name="cpi")
cpi_melted["coicop_code"] = cpi_melted["coicop_code"].astype("str")

# %%
# merged data
data = cpi_melted.merge(ei, on="coicop_code", how="inner").copy()

# %%
# plot coicop categories from 2015 onwards to see trends
ons_cat = ["#12436D", "#28A197", "#801650", "#F46A25", "#3D3D3D", "#A285D1"]
sns.set_palette(sns.color_palette(ons_cat))

data_plot = data[(data["date"] >= 201501)].copy()

# convert date string to integer from 1 to max
# needed as intervals between date strings might not be the same
data_plot["rank"] = data_plot["date"].rank(method="dense")
rank_treatment = data_plot[data_plot["date"] == 202202]["rank"].unique()[0]

fig, ax = plt.subplots()
sns.lineplot(
    data=data_plot, x="rank", y="cpi", hue="energy_intensity_group", ax=ax
)
plt.vlines(
    x=rank_treatment,
    ymin=ax.get_ylim()[0],
    ymax=ax.get_ylim()[1],
    linestyles=":",
    colors="black",
)
ax.set_xticks(list(data_plot["rank"].unique())[1::16])
ax.set_xticklabels([str(d) for d in list(data_plot["date"].unique())[1::16]])
plt.show()

# same but without energy
fig, ax = plt.subplots()
sns.lineplot(
    data=data_plot[data_plot["energy_intensity_group"] != "Energy"],
    x="rank",
    y="cpi",
    hue="energy_intensity_group",
    ci=False,
    ax=ax,
)
plt.vlines(
    x=rank_treatment,
    ymin=ax.get_ylim()[0],
    ymax=ax.get_ylim()[1],
    linestyles=":",
    colors="black",
)
ax.set_xticks(list(data_plot["rank"].unique())[1::16])
ax.set_xticklabels([str(d) for d in list(data_plot["date"].unique())[1::16]])
plt.savefig(os.path.join(here(), "outputs", "descriptive1.png"))
plt.show()

# %%
# by energy intensity category
# reduce time granularity as some categories don't update every month
data_plot = data[
    (data["date"] >= 201001)
    & (data["date"].astype("str").str[-2:].isin(["02", "08"]))
].copy()

# convert date string to integer from 1 to max
# needed as intervals between date strings might not be the same
data_plot["rank"] = data_plot["date"].rank(method="dense")
rank_treatment = data_plot[data_plot["date"] == 202202]["rank"].unique()[0]

fig, axs = plt.subplots(2, 3, sharey=True, sharex=True, figsize=(12, 8))
ei_group = list(data_plot["energy_intensity_group"].unique())

for ei, ax in zip(ei_group, axs.ravel()):
    sel = data_plot[data_plot["energy_intensity_group"] == ei]
    sns.lineplot(sel, x="rank", y="cpi", ax=ax)
    ax.set_title(ei)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.vlines(
        x=rank_treatment, ymin=250, ymax=50, linestyles=":", colors="black"
    )
plt.tight_layout()
plt.savefig(os.path.join(here(), "outputs", "descriptive2.png"))
plt.show()

# %%
# export processed data for analysis
data.to_csv(
    os.path.join(here(), "data", "processed", "processed_data.csv"),
    index=False,
)
# %%
