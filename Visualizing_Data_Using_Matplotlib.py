"""
Data Visualization with Matplotlib and Seaborn
Case Study: Understanding Bike Rental Patterns


"""

# %% 4.1.1 Imports
import io
import zipfile
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

sns.set_theme(style="whitegrid")  # consistent, clean default style

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# %% 4.1.2 Download the dataset (cached in ./data)
URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DAY_CSV = DATA_DIR / "day.csv"

if not DAY_CSV.exists():
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(DATA_DIR)
    print("Dataset downloaded and extracted successfully.")
else:
    print("Dataset already downloaded.")

df = pd.read_csv(DAY_CSV)
df["dteday"] = pd.to_datetime(df["dteday"])
print(df.head())

# %% 4.1.3 - 4.1.4 Structure and summary statistics
print("Dataset shape:", df.shape)
df.info()
print(df.describe())

# %% 4.1.5 Categorical variables stored as numbers
print("Unique seasons:", df["season"].unique())
print("Unique weather situations:", df["weathersit"].unique())
print("Working day values:", df["workingday"].unique())

# %% 4.1.7 / 4.2 First plot vs. a better plot
plt.plot(df["cnt"])
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df["cnt"])
plt.title("Daily Bike Rentals Over Time")
plt.xlabel("Day Index")
plt.ylabel("Total Rentals (cnt)")
plt.grid(True)
plt.show()

# %% 4.3.0 Layout and customization
plt.figure(figsize=(9, 5))
plt.scatter(df["temp"], df["cnt"], alpha=0.6)
plt.title("Temperature vs Total Rentals", fontsize=16)
plt.xlabel("Temperature", fontsize=13)
plt.ylabel("Total Rentals", fontsize=13)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.grid(True)
plt.show()

# %% 4.3.0 Subplots
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(df["cnt"])
plt.title("Total Rentals Over Time")
plt.ylabel("Total Rentals")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(df["temp"])
plt.title("Temperature Over Time")
plt.ylabel("Temperature")
plt.xlabel("Day Index")
plt.grid(True)

plt.tight_layout()
plt.show()

# %% 4.3.1 Line plot with a proper date axis
plt.figure(figsize=(12, 5))
plt.plot(df["dteday"], df["cnt"])
plt.title("Daily Bike Rentals Over Time")
plt.xlabel("Date")
plt.ylabel("Total Daily Rentals")
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% Multiple lines: casual vs registered
plt.figure(figsize=(12, 5))
plt.plot(df["dteday"], df["casual"], label="Casual Users")
plt.plot(df["dteday"], df["registered"], label="Registered Users")
plt.title("Casual vs Registered Bike Rentals Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Rentals")
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% Practice 4.3.1.1 - Monthly average rentals (solution)
df["year_month"] = df["dteday"].dt.to_period("M")
monthly_avg = df.groupby("year_month")["cnt"].mean()
monthly_avg.index = monthly_avg.index.to_timestamp()

plt.figure(figsize=(12, 5))
plt.plot(monthly_avg.index, monthly_avg.values, marker="o")
plt.title("Average Monthly Bike Rentals")
plt.xlabel("Month")
plt.ylabel("Average Daily Rentals")
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# %% Practice 4.3.1.2 - 7-day rolling average (solution)
df["rolling_7"] = df["cnt"].rolling(window=7).mean()

plt.figure(figsize=(12, 5))
plt.plot(df["dteday"], df["cnt"], alpha=0.4, label="Daily")
plt.plot(df["dteday"], df["rolling_7"], linewidth=2, label="7-day average")
plt.title("Daily Rentals with 7-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Total Rentals")
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# %% 4.3.2 Histograms
plt.figure(figsize=(8, 5))
plt.hist(df["cnt"], bins=20, edgecolor="black")
plt.title("Distribution of Daily Bike Rentals")
plt.xlabel("Total Daily Rentals")
plt.ylabel("Frequency")
plt.show()

# Custom bin edges (every 500 rentals)
bin_edges = np.arange(0, 9500, 500)
plt.figure(figsize=(8, 5))
plt.hist(df["cnt"], bins=bin_edges, edgecolor="black")
plt.title("Histogram of Daily Bike Rentals (Bins Every 500 Rentals)")
plt.xlabel("Total Daily Rentals")
plt.ylabel("Frequency")
plt.show()

# %% Practice 4.3.2.1 - 10 vs 30 bins (solution)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(df["cnt"], bins=10, edgecolor="black")
plt.title("10 Bins")
plt.subplot(1, 2, 2)
plt.hist(df["cnt"], bins=30, edgecolor="black")
plt.title("30 Bins")
plt.tight_layout()
plt.show()

# %% Practice 4.3.2.2 - temp vs cnt distributions (solution)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(df["temp"], bins=20, edgecolor="black")
plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.subplot(1, 2, 2)
plt.hist(df["cnt"], bins=20, edgecolor="black")
plt.title("Rental Distribution")
plt.xlabel("Total Rentals")
plt.tight_layout()
plt.show()

# %% Practice 4.3.2.3 - weekday vs weekend (solution)
weekday_rentals = df[df["workingday"] == 1]["cnt"]
weekend_rentals = df[df["workingday"] == 0]["cnt"]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(weekday_rentals, bins=20, edgecolor="black")
plt.title("Weekday Rentals")
plt.xlabel("Total Rentals")
plt.subplot(1, 2, 2)
plt.hist(weekend_rentals, bins=20, edgecolor="black")
plt.title("Weekend/Holiday Rentals")
plt.xlabel("Total Rentals")
plt.tight_layout()
plt.show()

# %% 4.3.3 Boxplot
plt.figure(figsize=(8, 3))
plt.boxplot(df["cnt"], vert=False)
plt.title("Boxplot of Daily Bike Rentals", fontsize=14)
plt.xlabel("Total Daily Rentals", fontsize=12)
plt.yticks([])
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.show()

# %% Practice 4.3.3.1 - temperature boxplot (solution)
plt.figure(figsize=(8, 3))
plt.boxplot(df["temp"], vert=False)
plt.title("Boxplot of Temperature")
plt.xlabel("Normalized Temperature")
plt.yticks([])
plt.show()

# %% Practice 4.3.3.2 - weekday vs weekend boxplots (solution)
weekday = df[df["workingday"] == 1]["cnt"]
weekend = df[df["workingday"] == 0]["cnt"]

plt.figure(figsize=(6, 5))
plt.boxplot([weekday, weekend], tick_labels=["Weekday", "Weekend/Holiday"])
plt.title("Rentals: Weekday vs Weekend")
plt.ylabel("Total Rentals")
plt.show()

# %% 4.4.1 Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(df["temp"], df["cnt"], alpha=0.6)
plt.title("Temperature vs Total Rentals", fontsize=14)
plt.xlabel("Normalized Temperature", fontsize=12)
plt.ylabel("Total Daily Rentals", fontsize=12)
plt.show()

# %% Practice 4.4.1.1 - humidity vs rentals (solution)
plt.figure(figsize=(8, 5))
plt.scatter(df["hum"], df["cnt"], alpha=0.6)
plt.title("Humidity vs Total Rentals")
plt.xlabel("Humidity")
plt.ylabel("Total Rentals")
plt.show()

# %% Practice 4.4.1.2 - temperature vs casual users (solution)
plt.figure(figsize=(8, 5))
plt.scatter(df["temp"], df["casual"], alpha=0.6)
plt.title("Temperature vs Casual Rentals")
plt.xlabel("Temperature")
plt.ylabel("Casual Rentals")
plt.show()

# %% 4.4.2 Bubble chart (size = wind speed)
plt.figure(figsize=(8, 5))
plt.scatter(df["temp"], df["cnt"], s=df["windspeed"] * 1000, alpha=0.4)
plt.title("Temperature vs Rentals (Bubble Size = Wind Speed)", fontsize=14)
plt.xlabel("Temperature", fontsize=12)
plt.ylabel("Total Rentals", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# %% Practice 4.4.2.1 - bubble size = humidity (solution)
plt.figure(figsize=(8, 5))
plt.scatter(df["temp"], df["cnt"], s=df["hum"] * 100, alpha=0.4)
plt.title("Temperature vs Rentals (Bubble Size = Humidity)")
plt.xlabel("Temperature")
plt.ylabel("Total Rentals")
plt.show()

# %% 4.5.1 Bar chart: average rentals by season
season_map = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Autumn"}

season_avg = (
    df.assign(Season=df["season"].map(season_map))
    .groupby("Season")["cnt"]
    .mean()
    .reindex(season_map.values())
)

plt.figure(figsize=(8, 5))
plt.bar(season_avg.index, season_avg.values)
plt.title("Average Bike Rentals by Season")
plt.xlabel("Season")
plt.ylabel("Average Rentals")
plt.show()

# Horizontal version
plt.figure(figsize=(8, 5))
plt.barh(season_avg.index[::-1], season_avg.values[::-1])
plt.title("Average Bike Rentals by Season")
plt.ylabel("Season")
plt.xlabel("Average Rentals")
plt.show()

# %% Practice 4.5.1.1 - average temperature by season (solution)
temp_avg = (
    df.assign(Season=df["season"].map(season_map))
    .groupby("Season")["temp"]
    .mean()
    .reindex(season_map.values())
)

plt.figure(figsize=(8, 5))
plt.bar(temp_avg.index, temp_avg.values)
plt.title("Average Temperature by Season")
plt.xlabel("Season")
plt.ylabel("Average Normalized Temperature")
plt.show()

print("Season with highest average temperature:", temp_avg.idxmax())

# %% Practice 4.5.1.2 - total rentals by month (solution)
monthly_total = df.groupby(df["dteday"].dt.month)["cnt"].sum()

plt.figure(figsize=(10, 5))
plt.bar(monthly_total.index, monthly_total.values)
plt.xticks(range(1, 13))
plt.title("Total Rentals by Month")
plt.xlabel("Month")
plt.ylabel("Total Rentals")
plt.show()

print("Month with highest total rentals:", monthly_total.idxmax())

# %% 4.5.2 Grouped bar chart (Matplotlib)
season_labels = ["Winter", "Spring", "Summer", "Autumn"]
casual_avg = df.groupby("season")["casual"].mean()
registered_avg = df.groupby("season")["registered"].mean()

x = np.arange(len(casual_avg))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, casual_avg.values, width, label="Casual")
plt.bar(x + width / 2, registered_avg.values, width, label="Registered")
plt.title("Average Rentals by Season", fontsize=14)
plt.xlabel("Season", fontsize=12)
plt.ylabel("Average Rentals", fontsize=12)
plt.xticks(x, season_labels)
plt.legend(title="User Type")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

# %% 4.5.3 Stacked bar chart
season_totals = df.groupby("season")[["casual", "registered"]].mean()
casual_vals = season_totals["casual"].values
registered_vals = season_totals["registered"].values
x = np.arange(len(season_totals))

plt.figure(figsize=(8, 5))
plt.bar(x, casual_vals, label="Casual")
plt.bar(x, registered_vals, bottom=casual_vals, label="Registered")
plt.title("Average Rentals by Season (Stacked)", fontsize=14)
plt.xlabel("Season", fontsize=12)
plt.ylabel("Average Rentals", fontsize=12)
plt.xticks(x, season_labels)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# %% Practice 4.5.3.1 - stacked bar by working day (solution)
workingday_avg = df.groupby("workingday")[["casual", "registered"]].mean()
casual_vals = workingday_avg["casual"].values
registered_vals = workingday_avg["registered"].values
x = np.arange(len(workingday_avg))

plt.figure(figsize=(8, 5))
plt.bar(x, casual_vals, label="Casual")
plt.bar(x, registered_vals, bottom=casual_vals, label="Registered")
plt.xticks(x, ["Non-working day", "Working day"])
plt.title("Average Rentals by Working Day (Stacked)")
plt.ylabel("Average Rentals")
plt.legend()
plt.tight_layout()
plt.show()

# %% 4.6 Multi-panel figures
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df["cnt"], bins=20)
plt.title("Distribution of Rentals")
plt.xlabel("Total Rentals")
plt.ylabel("Frequency")
plt.subplot(1, 2, 2)
plt.hist(df["temp"], bins=20)
plt.title("Distribution of Temperature")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 4))
for i, (col, label) in enumerate(
    [("temp", "Temperature"), ("hum", "Humidity"), ("windspeed", "Wind Speed")], start=1
):
    plt.subplot(1, 3, i)
    plt.scatter(df[col], df["cnt"], alpha=0.6)
    plt.title(f"{label} vs Rentals")
    plt.xlabel(label)
    plt.ylabel("Rentals")
plt.tight_layout()
plt.show()

# %% 4.7 Seaborn: readable labels first
df["Season"] = df["season"].map(season_map)  # fixed "Autum" typo from the original
df["Working Day"] = df["workingday"].map({0: "No", 1: "Yes"})

# Grouped bar chart with hue
plt.figure(figsize=(9, 5))
sns.barplot(data=df, x="Working Day", y="cnt", hue="Season", estimator="mean", errorbar=None)
plt.title("Average Rentals by Working Day and Season")
plt.xlabel("Working Day")
plt.ylabel("Average Rentals")
plt.tight_layout()
plt.show()

# %% 4.7.3 Scatter plot with hue
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="temp", y="cnt", hue="Season", alpha=0.6)
plt.title("Temperature vs Rentals by Season")
plt.xlabel("Temperature")
plt.ylabel("Total Rentals")
plt.tight_layout()
plt.show()

# %% 4.7.4 Boxplot with hue
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="weathersit", y="cnt", hue="Season")
plt.title("Rental Distribution by Weather and Season")
plt.xlabel("Weather Category")
plt.ylabel("Total Rentals")
plt.tight_layout()
plt.show()

# %% 4.8 Correlation matrix and heatmap
numeric_df = df[["temp", "hum", "windspeed", "casual", "registered", "cnt"]]
corr_matrix = numeric_df.corr()
print(corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", center=0, square=True, linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

# %% 4.9 Regression
plt.figure(figsize=(8, 5))
sns.regplot(data=df, x="temp", y="cnt", scatter_kws={"alpha": 0.5})
plt.title("Regression: Temperature vs Total Rentals")
plt.xlabel("Temperature")
plt.ylabel("Total Rentals")
plt.tight_layout()
plt.show()

# lmplot creates its own figure, so title goes on the FacetGrid
g = sns.lmplot(data=df, x="temp", y="cnt", hue="Season", height=6, aspect=1.3, scatter_kws={"alpha": 0.4})
g.figure.suptitle("Temperature vs Rentals by Season", y=1.02)
plt.show()

# =====================================================================
# Appendix
# =====================================================================

# %% A-1 Plotting function graphs
def my_function(x):
    return x**2 + 2 * x + 1


x = np.arange(-10, 10)
plt.figure(figsize=(16, 4))
plt.plot(x, my_function(x))
plt.grid(True)
plt.show()

# Practice 1: y = 5x + 3
x = np.linspace(-10, 10, 100)
y = 5 * x + 3
plt.plot(x, y)
plt.title("y = 5x + 3")
plt.grid(True)
plt.show()

# Practice 2: sin and cos
plt.plot(x, np.sin(x), label="sin(x)")
plt.plot(x, np.cos(x), label="cos(x)")
plt.legend()
plt.grid(True)
plt.show()

# %% A-2 KDE plot (the original used an undefined `iris`; load it from seaborn)
iris = sns.load_dataset("iris")  # downloads a small CSV on first use

plt.figure(figsize=(12, 4))
sns.histplot(data=iris, x="sepal_length", hue="species", bins=25, binrange=(4.5, 7.5), kde=True)
plt.xlabel("Sepal Length")
plt.show()

# %% A-3 Financial data: random walk + candlestick chart (needs plotly)
idx = pd.date_range("2015-01-01", "2015-12-31 23:59", freq="min")  # "T" is deprecated
dn = np.random.randint(2, size=len(idx)) * 2 - 1
rnd_walk = np.cumprod(np.exp(dn * 0.0002)) * 100
ohlc = pd.Series(rnd_walk, index=idx).resample("B").ohlc()

ohlc.plot(figsize=(15, 6), legend=True, grid=True)
plt.show()

import plotly.graph_objects as go  # noqa: E402

fig = go.Figure(
    data=[
        go.Candlestick(
            x=ohlc.index,
            open=ohlc["open"],
            high=ohlc["high"],
            low=ohlc["low"],
            close=ohlc["close"],
        )
    ]
)
fig.show()  # opens in your browser

# %% A-4 Automated EDA (optional: pip install ydata-profiling)
# The old pandas-profiling package is now called ydata-profiling.
# from sklearn.datasets import fetch_california_housing
# from ydata_profiling import ProfileReport
#
# ch = fetch_california_housing(as_frame=True).frame
# ProfileReport(ch, title="California Housing EDA").to_file("california_report.html")

# %% Bonus: Monte Carlo estimate of pi
import math  # noqa: E402

rng = np.random.default_rng(42)
n = 10_000
xs = rng.uniform(0.0, 1.0, n)
ys = rng.uniform(0.0, 1.0, n)

inside = np.array([math.hypot(a, b) < 1 for a, b in zip(xs, ys)])
pi_estimate = 4 * inside.sum() / n
print(f"Points inside circle: {inside.sum()} / {n}")
print(f"Estimated pi: {pi_estimate:.4f}")

plt.figure(figsize=(6, 6))
plt.scatter(xs[inside], ys[inside], s=2, label="Inside")
plt.scatter(xs[~inside], ys[~inside], s=2, label="Outside")
plt.gca().set_aspect("equal")
plt.title(f"Monte Carlo pi ≈ {pi_estimate:.4f}")
plt.legend(markerscale=5)
plt.show()