# Telecom Customer Churn Prediction with PySpark MLlib

## Project Overview

This project applies machine learning with PySpark to predict customer churn for a telecommunications company. The objective is to classify whether a customer is likely to churn based on account details, subscribed services, billing behavior, tenure, and charge-related attributes.

The workflow uses Apache Spark for distributed data processing and Spark MLlib for feature engineering, model training, and evaluation. A decision tree classifier is trained on a cleaned and transformed version of the telecom customer dataset, then evaluated with Area Under the ROC Curve (AUC) to measure binary classification performance.

The project also connects model outputs and exploratory analysis to a business recommendation: customers on month-to-month contracts show substantially higher churn than customers on one-year or two-year contracts, making contract migration incentives a practical retention strategy.

## Dataset Description

The dataset used in this project is `dataset.csv`, a telecom customer churn dataset with 7,043 original customer records and 21 columns. Each row represents one customer, and the target variable is `Churn`, which indicates whether the customer left the company.

Key dataset groups:

| Category | Columns |
| --- | --- |
| Customer identifiers and demographics | `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account tenure and billing | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` |
| Phone services | `PhoneService`, `MultipleLines` |
| Internet services | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Prediction label | `Churn` |

The Spark schema inferred three numerical columns:

| Column | Type | Description |
| --- | --- | --- |
| `tenure` | Integer | Number of months the customer has stayed with the company |
| `MonthlyCharges` | Double | Monthly amount charged to the customer |
| `TotalCharges` | Double | Total amount charged to the customer |

All remaining fields were loaded as categorical string columns. The binary label `Churn` contains two classes: `No` and `Yes`.

## Technologies Used

- Python
- PySpark
- Apache Spark SQL
- Spark MLlib
- pandas
- Matplotlib
- Plotly Express
- Jupyter Notebook

The notebook uses PySpark modules for data processing, feature transformation, model training, and evaluation:

- `SparkSession` for creating the Spark application context.
- `pyspark.sql.functions` for null counting and column-level transformations.
- `Imputer`, `StringIndexer`, `VectorAssembler`, and `StandardScaler` for preprocessing and feature preparation.
- `DecisionTreeClassifier` for churn classification.
- `BinaryClassificationEvaluator` for AUC-based model evaluation.

## Project Workflow

1. Install and import the required PySpark, visualization, and data analysis libraries.
2. Create a Spark session with the application name `Customer_Churn_Prediction`.
3. Load `dataset.csv` as a Spark DataFrame using CSV format, header parsing, and schema inference.
4. Inspect the schema, row count, column count, and initial records.
5. Separate numerical columns from categorical columns based on Spark-inferred data types.
6. Perform exploratory data analysis on numerical features using pandas, histograms, descriptive statistics, and correlation analysis.
7. Analyze categorical feature distributions with grouped counts for each string column.
8. Check null values across all columns and identify `TotalCharges` as the only column with missing values.
9. Impute missing `TotalCharges` values using Spark MLlib's `Imputer` with the mean strategy.
10. Detect and remove the single tenure outlier where `tenure > 100`, reducing the modeling dataset from 7,043 rows to 7,042 rows.
11. Assemble numerical features into `numerical_features_vector` using `VectorAssembler`.
12. Standardize numerical features into `numerical_features_scaled` with `StandardScaler` using both mean centering and standard deviation scaling.
13. Convert categorical string columns into numeric index columns using `StringIndexer`.
14. Remove `customerID_Indexed` from model features because it is an identifier, and remove `Churn_Indexed` from model features because it is the label.
15. Assemble the remaining indexed categorical columns into `categorical_features_vector`.
16. Combine `categorical_features_vector` and `numerical_features_scaled` into `final_feature_vector`.
17. Split the transformed dataset into training and test sets using a 70/30 split with seed `100`.
18. Train a `DecisionTreeClassifier` using `final_feature_vector` as the feature column and `Churn_Indexed` as the label column.
19. Generate predictions on the test set.
20. Evaluate model performance using Area Under the ROC Curve on both test and training data.
21. Compare multiple decision tree depths from `maxDepth=2` through `maxDepth=20`.
22. Use feature importance and contract-level churn analysis to form a retention recommendation.

## Exploratory Data Analysis

The notebook begins by validating the loaded Spark DataFrame:

- Original rows: `7,043`
- Original columns: `21`
- Numerical columns: `tenure`, `MonthlyCharges`, `TotalCharges`
- Categorical columns: customer, service, account, billing, and churn fields

Numerical analysis showed that customer tenure had a wide range before preprocessing. The descriptive statistics for `tenure` identified a maximum value of `458`, which is unusually high for a monthly telecom tenure field and was later treated as an outlier.

The numerical correlation matrix showed strong relationships between tenure and accumulated charges:

| Feature Pair | Correlation |
| --- | ---: |
| `tenure` and `TotalCharges` | `0.806530` |
| `MonthlyCharges` and `TotalCharges` | `0.651065` |
| `tenure` and `MonthlyCharges` | `0.243703` |

Categorical analysis was performed by grouping and counting unique values for every string column. The target distribution in the original dataset was:

| Churn | Count |
| --- | ---: |
| `No` | `5,174` |
| `Yes` | `1,869` |

Contract type was especially important for churn interpretation. After preprocessing and outlier removal, churn by contract type was:

| Contract Type | Non-Churned Customers | Churned Customers | Churn Rate |
| --- | ---: | ---: | ---: |
| `Month-to-month` | `2,219` | `1,655` | `42.72%` |
| `One year` | `1,307` | `166` | `11.27%` |
| `Two year` | `1,647` | `48` | `2.83%` |

This analysis shows that month-to-month customers account for the majority of churned customers and have a much higher churn rate than customers on longer contracts.

## Data Preprocessing and Feature Engineering

The preprocessing stage focused on missing-value handling, outlier removal, and converting raw columns into Spark MLlib-compatible feature vectors.

Missing-value handling:

- `TotalCharges` contained 11 missing values.
- A Spark MLlib `Imputer` was configured with `inputCols=["TotalCharges"]` and `outputCols=["TotalCharges"]`.
- The imputation strategy was set to `mean`.
- After transformation, the null count for `TotalCharges` was verified as `0`.

Outlier handling:

- The notebook searched for customers with `tenure > 100`.
- One customer record had `tenure = 458`.
- That row was removed with `data.filter(data.tenure < 100)`.
- Row count changed from `7,043` to `7,042`.

Numerical feature preparation:

- `tenure`, `MonthlyCharges`, and `TotalCharges` were assembled into `numerical_features_vector`.
- `StandardScaler` transformed this vector into `numerical_features_scaled`.
- Scaling used `withStd=True` and `withMean=True`, meaning each numerical feature was standardized using standard deviation scaling and mean centering.

Categorical feature preparation:

- All string columns were initially indexed using `StringIndexer`.
- `customerID_Indexed` was excluded from the feature vector because customer identifiers do not provide reusable predictive structure.
- `Churn_Indexed` was excluded from the feature vector and retained as the supervised learning label.
- The remaining indexed categorical variables were assembled into `categorical_features_vector`.

Final feature vector:

- `categorical_features_vector` contains 16 indexed categorical feature columns.
- `numerical_features_scaled` contains 3 scaled numerical features.
- These were combined into `final_feature_vector`, producing a 19-feature modeling input for Spark MLlib.

## Model Training

The transformed dataset was split into training and test subsets using:

- Split ratio: `70%` training, `30%` testing
- Random seed: `100`
- Training rows: `4,930`
- Test rows: `2,112`

The classifier used in the notebook was Spark MLlib's `DecisionTreeClassifier`.

Model configuration:

| Parameter | Value |
| --- | --- |
| Algorithm | `DecisionTreeClassifier` |
| Feature column | `final_feature_vector` |
| Label column | `Churn_Indexed` |
| Maximum depth | `6` |

The trained model generated predictions on the test set, returning both the original `Churn` value and the numeric `prediction` class.

## Model Evaluation

Model performance was evaluated with `BinaryClassificationEvaluator` using Area Under the ROC Curve.

| Dataset | AUC |
| --- | ---: |
| Test set | `0.7968240892739675` |
| Training set | `0.797607974377661` |

The close train and test AUC values indicate that the selected decision tree depth produced similar behavior on both seen and unseen data for this split.

The notebook also evaluated decision tree depth values from `2` through `20` to compare training and test AUC. The selected `maxDepth=6` produced the best test AUC among the evaluated depths.

| `maxDepth` | Training AUC | Test AUC |
| ---: | ---: | ---: |
| `2` | `0.7737711024910238` | `0.7918744332963926` |
| `3` | `0.7721112330375414` | `0.7678230877272001` |
| `4` | `0.6732017750857608` | `0.6938514510575503` |
| `5` | `0.5647403286677019` | `0.5664944085244075` |
| `6` | `0.797607974377661` | `0.7968240892739675` |
| `7` | `0.7600331160893413` | `0.7464077658395512` |
| `8` | `0.7616399322585649` | `0.7486976928866473` |
| `9` | `0.7851198149534798` | `0.7390283218461648` |
| `10` | `0.82591977092655` | `0.7568262227463716` |
| `11` | `0.8633625692715379` | `0.7717528253693575` |
| `12` | `0.8899907318309699` | `0.770466222983424` |
| `13` | `0.9236516335825413` | `0.7493673661690539` |
| `14` | `0.9402616892332477` | `0.7371905724224987` |
| `15` | `0.9521037242780167` | `0.7208594338001293` |
| `16` | `0.9662424766482165` | `0.7102797812005524` |
| `17` | `0.9763120733848871` | `0.704298946894316` |
| `18` | `0.9807472862567603` | `0.7036452746549405` |
| `19` | `0.9853808496196841` | `0.6956814962753128` |
| `20` | `0.9883361989750398` | `0.6895714683623821` |

As tree depth increases beyond the selected configuration, training AUC generally rises while test AUC declines, which suggests increasing model complexity and reduced generalization on the held-out data.

## Key Insights and Recommendation

The strongest business insight from the notebook is the relationship between contract type and churn. Customers with month-to-month contracts churn at a much higher rate than customers with one-year or two-year contracts.

Observed churn rates after preprocessing:

- Month-to-month: `42.72%`
- One year: `11.27%`
- Two year: `2.83%`

This pattern suggests that short-term contract customers are less committed and more likely to leave. A practical retention strategy is to encourage month-to-month customers to switch to longer-term contracts by offering incentives, loyalty discounts, bundled service benefits, or contract upgrade promotions.

The decision tree model supports churn risk prediction at the customer level, while the contract analysis translates the model exploration into an actionable retention policy.

## Conclusion

This project demonstrates an end-to-end churn prediction workflow using PySpark and Spark MLlib. The notebook loads and profiles telecom customer data, handles missing values and outliers, prepares numerical and categorical features, trains a decision tree classifier, evaluates model performance with AUC, and derives a business recommendation from churn patterns.

The final model uses a 19-feature input vector and achieves a test AUC of `0.7968240892739675`. Combined with the exploratory finding that month-to-month customers have the highest churn rate, the project provides both a predictive modeling approach and a clear customer retention recommendation.
