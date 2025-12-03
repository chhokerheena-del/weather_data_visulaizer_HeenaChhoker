import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Creating 'plots' folder if it does not exist
if not os.path.exists("plots"):
    os.makedirs("plots")

print("\n--- TASK 1: DATA LOADING ---\n")

# Load the real weather CSV file
df = pd.read_csv("weather_data.csv")

print("First 5 rows:\n", df.head())
print("\nInfo:\n")
print(df.info())
print("\nDescribe:\n", df.describe())


print("\n--- TASK 2: DATA CLEANING ---\n")

# Drop missing values
df = df.dropna()

# Convert date column
df["date"] = pd.to_datetime(df["date"], errors='coerce')

# Select important columns
required_cols = ["date", "temperature", "rainfall", "humidity"]
df = df[required_cols]

print("Cleaned data sample:\n", df.head())


print("\n--- TASK 3: NUMPY STATISTICS ---\n")

stats = df[["temperature", "rainfall", "humidity"]].agg(["mean", "min", "max", "std"])
print("Daily summary stats:\n", stats)

# Monthly grouping
df["month"] = df["date"].dt.month
monthly_stats = df.groupby("month")[["temperature", "rainfall", "humidity"]].mean()
print("\nMonthly Summary:\n", monthly_stats)


print("\n--- TASK 4: VISUALIZATIONS ---\n")

# Line plot
plt.figure(figsize=(9,5))
plt.plot(df["date"], df["temperature"])
plt.title("Daily Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.tight_layout()
plt.savefig("plots/temp_trend.png")
plt.close()

# Bar chart
plt.figure(figsize=(9,5))
plt.bar(monthly_stats.index, monthly_stats["rainfall"])
plt.title("Monthly Rainfall")
plt.xlabel("Month")
plt.ylabel("Rainfall")
plt.tight_layout()
plt.savefig("plots/rainfall_monthly.png")
plt.close()

# Scatter plot
plt.figure(figsize=(9,5))
plt.scatter(df["temperature"], df["humidity"])
plt.title("Humidity vs Temperature")
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.tight_layout()
plt.savefig("plots/humidity_vs_temp.png")
plt.close()

# Combined
plt.figure(figsize=(10,5))
plt.plot(df["date"], df["temperature"], label="Temperature")
plt.plot(df["date"], df["humidity"], label="Humidity")
plt.xlabel("Date")
plt.title("Temperature & Humidity")
plt.legend()
plt.tight_layout()
plt.savefig("plots/combined_plot.png")
plt.close()


print("\n--- TASK 5: GROUPING & AGGREGATION ---\n")

season_map = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn"
}

df["season"] = df["month"].map(season_map)
seasonal_stats = df.groupby("season")[["temperature","rainfall","humidity"]].mean()
print(seasonal_stats)


print("\n--- TASK 6: EXPORT ---\n")

df.to_csv("cleaned_weather_data.csv", index=False)
print("Cleaned CSV saved as cleaned_weather_data.csv")
print("Plots saved in 'plots' folder.")

print("\nProject Completed Successfully!")
