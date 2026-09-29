import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {
    "title": [
        "Avatar", "Titanic", "Inception", "The Dark Knight",
        "Interstellar", "Gladiator", "The Matrix",
        "Parasite", "Avengers Endgame", "Toy Story"
    ],
    "genres": [
        "Action|Adventure", "Drama|Romance", "Sci-Fi|Thriller",
        "Action|Crime", "Sci-Fi|Drama", "Action|Drama",
        "Sci-Fi|Action", "Drama|Thriller", "Action|Adventure",
        "Animation|Comedy"
    ],
    "year": [2009, 1997, 2010, 2008, 2014, 2000, 1999, 2019, 2019, 1995],
    "rating": [7.8, 7.9, 8.8, 9.0, 8.7, 8.5, 8.7, 8.5, 8.4, 8.3]
}

df = pd.DataFrame(data)

df.to_csv("movies_cleaned.csv", index=False)

print("===== DATASET OVERVIEW =====")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n===== BASIC STATISTICS =====")
print(df[["year", "rating"]].describe().round(2))

print("\n===== HIGHEST RATED MOVIES =====")
highest = df.nlargest(5, "rating")
print(highest[["title", "year", "rating"]].to_string(index=False))

print("\n===== LOWEST RATED MOVIES =====")
lowest = df.nsmallest(5, "rating")
print(lowest[["title", "year", "rating"]].to_string(index=False))

print("\n===== AVERAGE RATING BY GENRE =====")
genre_data = df.assign(genres=df["genres"].str.split("|")).explode("genres")
genre_ratings = genre_data.groupby("genres")["rating"].mean().sort_values(ascending=False)
print(genre_ratings.round(2))

print("\n===== MOVIES BY YEAR =====")
year_counts = df.groupby("year").size().sort_values(ascending=False)
print(year_counts)

print("\n===== KEY FINDINGS =====")
print("1. The highest rated movie is:", highest.iloc[0]["title"])
print("2. The lowest rated movie is:", lowest.iloc[0]["title"])
print("3. The average movie rating is:", round(df["rating"].mean(), 2))
print("4. The highest rating is:", df["rating"].max())
print("5. The lowest rating is:", df["rating"].min())

correlation = df["year"].corr(df["rating"])
print("6. Correlation between release year and rating:", round(correlation, 2))

plt.figure(figsize=(8, 5))
plt.hist(df["rating"], bins=5, edgecolor="black")
plt.title("Movie Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Movies")
plt.tight_layout()
plt.savefig("rating_distribution.png")
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df["year"], df["rating"])
plt.title("Release Year vs Movie Rating")
plt.xlabel("Release Year")
plt.ylabel("Rating")
plt.tight_layout()
plt.savefig("year_rating_relationship.png")
plt.show()