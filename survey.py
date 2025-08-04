# # Australian Students Survey Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("survey.csv")
df.head()


#  1. How many students were surveyed?

len(df)

#  2. Facts recorded and types (quantitative vs qualitative)

quantitative = df.select_dtypes(include="number").columns.tolist()
qualitative = df.select_dtypes(include="object").columns.tolist()
quantitative, qualitative


#  3. Did all students provide all facts?

df.isnull().sum()


#  4. Balance in terms of sex

df["Sex"].value_counts(dropna=False)


#  5. Representation of left-handers, smokers, and exercisers

left_handed = df["W.Hnd"].value_counts(dropna=False)
smoking = df["Smoke"].value_counts(dropna=False)
exercise = df["Exer"].value_counts(dropna=False)
left_handed, smoking, exercise

# by percentage:

left_handed = (df["W.Hnd"] == "Left").sum() / len(df) * 100
non_smokers = (df["Smoke"] == "Never").sum() / len(df) * 100
regular_exercisers = (df["Exer"] == "Freq").sum() / len(df) * 100
non_exercisers = df["Exer"].isna().sum() / len(df) * 100
left_handed, non_smokers, regular_exercisers, non_exercisers


#  6. Average height and pulse, and spread

height_stats = df["Height"].describe()
pulse_stats = df["Pulse"].describe()
height_stats, pulse_stats


#  7. Boxplots of height and pulse by exercise level

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(data=df, x="Exer", y="Height", ax=axs[0])
axs[0].set_title("Height vs Exercise")
sns.boxplot(data=df, x="Exer", y="Pulse", ax=axs[1])
axs[1].set_title("Pulse vs Exercise")
plt.tight_layout()
plt.show()


# 8. Boxplot of pulse by smoking habit

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Smoke", y="Pulse")
plt.title("Pulse vs Smoking Habit")
plt.show()


#  9. Scatter plots of hand spans and height

sns.pairplot(df, vars=["Wr.Hnd", "NW.Hnd", "Height"])
plt.show()
