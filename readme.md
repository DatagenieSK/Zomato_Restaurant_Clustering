# 🍽️ Zomato Restaurant Clustering & Sentiment Analysis

### Understanding restaurants, customer preferences and review patterns using Machine Learning.

This project started with a simple question: **Can we group restaurants based on more than just their ratings?**

A restaurant's price, cuisine, customer reviews, popularity and overall experience all tell a different part of its story. I wanted to bring these pieces together and see whether machine learning could uncover meaningful restaurant segments.

Using Zomato restaurant data from Hyderabad, I explored customer behaviour, performed sentiment analysis on reviews and built multiple clustering models to identify restaurants with similar characteristics.

---

## 📌 Project Overview

| Details | Information |
|---|---|
| Project Type | Machine Learning & NLP |
| Domain | FoodTech / Restaurant Analytics |
| Contribution | Individual |
| Author | SK Alauddin |
| Dataset | Zomato Hyderabad |
| Main Techniques | Clustering, Sentiment Analysis, NLP |

## 🎯 What I wanted to achieve

The project focuses on answering a few practical business questions:

- Does spending more money actually mean getting a better restaurant experience?
- What are customers talking about in positive and negative reviews?
- Do reviews with pictures receive different ratings?
- Can restaurants be grouped into meaningful price and experience categories?
- How can Zomato use these segments to improve recommendations and marketing strategies?

## 📂 Dataset

I worked with two datasets:

**1. Restaurant Metadata**
- 105 restaurants
- Restaurant names and Zomato links
- Cost information
- Collections and categories
- Cuisines
- Operating timings

**2. Customer Reviews**
- 10,000 customer reviews
- Restaurant and reviewer information
- Review text and ratings
- Reviewer followers
- Pictures posted
- Review timestamps

After cleaning and preprocessing, the analysis covered 100 restaurants with review data.

## 🛠️ Technologies & Libraries

- **Python** for data analysis and model development
- **Pandas & NumPy** for data manipulation
- **Matplotlib & Seaborn** for visualisation
- **Scikit-learn** for machine learning
- **NLTK** for text preprocessing and lemmatization
- **TF-IDF** for text vectorisation
- **SciPy** for statistical analysis and hierarchical clustering
- **Joblib** for model persistence

## 🔍 Project Workflow

### 1. Data Cleaning & Exploration
- Checked missing values and duplicate records.
- Cleaned restaurant cost and timing information.
- Processed reviewer metadata.
- Explored restaurant ratings, cuisines and customer behaviour.
- Performed univariate, bivariate and multivariate analysis.

### 2. Exploratory Data Analysis (EDA)

I explored relationships between:
- Restaurant cost and average ratings.
- Customer ratings and review engagement.
- Reviewer followers and restaurant popularity.
- Picture uploads and customer ratings.
- Review timings and customer activity.

### 3. Statistical Hypothesis Testing

Used statistical tests to investigate relationships in the data.

One interesting finding was that restaurant cost and average rating had a moderate positive correlation.

**Spearman correlation:**
- Correlation coefficient: 0.383
- P-value: 0.0001

This suggests that more expensive restaurants tend to have higher ratings, although price alone does not guarantee customer satisfaction.

### 4. NLP & Sentiment Analysis

Customer reviews contain information that numerical ratings cannot fully explain.

I performed text preprocessing, including:
- Expanding contractions.
- Removing unwanted characters and noise.
- Tokenisation.
- Stopword removal while retaining negations.
- Lemmatization.
- TF-IDF vectorisation using unigrams and bigrams.

I then trained a Logistic Regression classifier to identify positive, negative and neutral reviews.

**Sentiment model performance:**
- Test accuracy: 80%
- Macro F1-score: 0.70

The sentiment scores showed a strong correlation with average restaurant ratings (r = 0.97).

### 5. Restaurant Clustering

This was the main machine learning component of the project.

I experimented with three unsupervised learning algorithms:

1. **K-Means Clustering**
2. **Hierarchical (Agglomerative) Clustering**
3. **Gaussian Mixture Model (GMM)**

I compared the models using:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Bootstrap stability analysis

After experimentation and hyperparameter tuning, I selected a Gaussian Mixture Model with spherical covariance and three components.

## 📊 Model Performance

| Model | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|---:|---:|---:|
| K-Means | 0.277 | 1.221 | 41.1 |
| Hierarchical | 0.262 | 1.286 | 38.3 |
| Gaussian Mixture (Baseline) | 0.219 | 1.494 | 26.5 |
| **Tuned GMM** | **0.282** | **1.196** | **40.2** |

The tuned GMM performed best on silhouette and Davies-Bouldin scores while maintaining actionable segment sizes.

## 💡 Key Findings: Three Restaurant Segments

The final model identified three interesting restaurant groups.

### 1. Premium Crowd-Pleasers
- 13 restaurants
- Average cost for two: approximately ₹1,350
- Average rating: 4.34
- Negative reviews: 10%

These restaurants generally offer higher customer satisfaction, stronger ratings and better Zomato collection visibility.

### 2. Buzzy Mid-Premium
- 34 restaurants
- Average cost for two: approximately ₹1,190
- Average rating: 3.70
- Negative reviews: 19%

This segment attracts highly engaged reviewers, longer reviews and more picture uploads.

### 3. Budget Quick-Service
- 53 restaurants
- Average cost for two: approximately ₹550
- Average rating: 3.36
- Negative reviews: 32%

These restaurants are more budget-friendly, have longer operating hours and receive a higher proportion of negative reviews.

## 📈 Business Applications

The interesting part is how these findings can actually be used.

**For Zomato:**
- Build more relevant restaurant recommendations.
- Run targeted promotional campaigns.
- Identify restaurants that need service improvements.
- Improve restaurant visibility through collection placement.
- Understand customer preferences across price segments.

**For Restaurant Owners:**
- Compare performance with similar restaurants.
- Identify recurring customer complaints.
- Understand what influences customer satisfaction.
- Make better decisions about pricing, service and marketing.

## ⚙️ How to Run the Project

The notebook was designed to run in Google Colab.

**Step 1: Clone the repository**

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

**Step 2: Install dependencies**

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn joblib ipython nltk
```

**Step 3: Upload the datasets**

Upload these two CSV files to the Google Colab Files panel:

```text
Zomato Restaurant names and Metadata.csv
Zomato Restaurant reviews.csv
```

**Step 4: Open and run the notebook**

Open `Zomato_Restaurant_Clustering.ipynb` in Google Colab and execute all cells from top to bottom.

The notebook uses a fixed random seed (`42`) to support reproducibility.

## 🚀 Future Improvements

There is plenty of room to take this project further.

Some improvements I would like to explore:

- Expand the analysis to restaurants across multiple cities.
- Use transformer-based models for more advanced sentiment analysis.
- Include restaurant location, delivery time and order-volume data.
- Build an interactive dashboard using Streamlit or Power BI.
- Deploy the clustering model through a web application.
- Retrain the model periodically using fresh customer reviews.

## ⚠️ Limitations

- The analysis focuses only on Hyderabad restaurants.
- The clustering dataset contains 100 restaurants, which is relatively small.
- The restaurant review sample contains exactly 100 reviews per restaurant.
- The clusters overlap to some extent, so segment membership should not be treated as an absolute classification.
- Sentiment classification uses review ratings as a proxy for sentiment labels.

## 👨‍💻 About the Project

This project was an opportunity to work through a complete machine learning workflow, from understanding raw data to building and comparing models.

More importantly, it helped me understand that machine learning is not just about achieving a good evaluation score. The real value comes from turning data into insights that people can actually use.

**Developed by SK Alauddin**

If you find this project interesting, feel free to explore the notebook, share your feedback or connect with me!

⭐ If you like the project, consider giving this repository a star.

---

*Made with Python, curiosity and a lot of data cleaning.*
