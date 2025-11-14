# Team 7 

## Content

* [Purpose & Overview](#purpose--overview)
* [Goals & Objectives](#goals--objectives)
* [Techniques & Technologies](#techniques--technologies)
* [Key Findings & Instructions](#key-findings--instructions)
* [Visuals & Credits](#visuals--credits)

## Purpose & Overview
This project focuses on determing which variables significantly predict the occurence of stroke.

**Business Problem:**
We are a data science team working on stroke prevention. Our job is to advise public health decision makers on which factors significantly predict the occurence of stroke and use that to inform the population at risk who should receive stroke prevention treatment.

Our dataset consists of 11 variables for 5,110 patients, including the variable "stroke" which is 1 if a patient had a stroke or 0 otherwise. The remaining variables represent a patient's demographic, health, and lifestyle information.

* Age: age of patient
* Gender: gender of patient
* Hypertension: whether the patient has a hypertension
* Heart Disease: whether the patient has heart disease
* Ever Married: whether the patient is married
* Work Type: the type of employment the patient has
* Residence Type: the type of area where the patient resides
* Average Glucose Level: the patient's average glucose level
* BMI: the body mass index of the patient
* Smoking Status: the patient's smoking history/status

### Exploratory data analysis
#### Class distribution of stroke

  This bar plot shows the distribution of the target variable 'Stroke' across the dataset. As observed, the class distribution is highly imbalanced, where out of 5110 individuals, 4861 (95.13%) are labeled non-stroke and 249 (4.87%) are labeled stroke.

![alt text](images/distribution.png)

* Comparing stroke rates between gender: 

    It is observed that the occurence of stroke in both males and females are almost same - 4.71% for females and 5.11% for males.

* Stroke based on marital status:

    From the data, occurrence among married individuals is 6.56% and among unmarried individuals is 1.65%.
    
    Considering all patients who had stroke, 88.35% of them are married and 11.84% are unmarried, as demonstrated in the pie chart. This indicates a potential correlation between marital status and stroke.

    ![alt text](images/em.png)

* Proportion of stroke by residence type:

  From the data, 54.22% of the individuals who had a stroke are from urban areas and 45.78% are from rural areas.

* Proportion of stroke by work type:

  The distribution of stroke cases based on work type shows that among the patients who suffered a stroke, 59.84% work private, 26.10% are self-employed, 13.25% are government job holders and very small percentages in the other categories 'children' and 'never worked'.

  ![alt text](images/worktype.png)

* Stroke by smoking status:

  From the smoking status distribution it is observed that among the patients who suffered a stroke, 36.14% never smoked, 28.11% formerly smoked, 16.87% currently smokes, and 18.88% have unknown smoking status.

  ![alt text](images/smoke.png)

**Industry Context:**
In 2021, stroke was one of the top 5 leading causes of death in Canada, responsible for 37 deaths per 100,000 people. Being able to identify predictors of stroke plays a critical role in stroke prevention for the healthcare industry.

1. Early Detection: 
The ability to predict a stroke before it happens leads to more opportunities to prevent the stroke through lifestyle changes and prevention treatments.
      
2. Targeted Treatment:
Identifying which factors predict stroke aides healthcare professionals with developing treatments and interventions for strokes.

## Goals & Objectives
The project aims to develop a reliable model to predict stroke and identify associated key features. Our goal is to create a model with high performance metrics.

Initially we aimed to optimize for F1, because it is a balanced metric of precision and recall, but due to class imbalance in the dataset we were unable to achieve a high F1 score (max 30%), so we pivoted our modelling approach to optimize for recall instead of F1. The rationale for a high recall model is that false positives (predicting a stroke for a non-stroke patient) are less harmful than false negatives (predicting no stroke for a patient that suffers a stroke).

| Metrics for predicting stroke = 1 (Optimized for F1) | Logistic Regression | XGBoost Model |
|----------|----------|----------|
| Accuracy | 0.85 | 0.84 |
| Precision | 0.20 | 0.19 |
| F1 Score | 0.32 | 0.30 |
| Recall | 0.74 | 0.70 |
| ROC AUC | 0.84 | 0.85 |

## Techniques & Technologies
The data used for this project was the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?resource=download).

Analyses were performed using Python 3.9.15 in Jupyter notebooks.

The libraries used in this project were pandas, numpy, scikit-learn, shap, statsmodels, matplotlib, xgboost, seaborn, plotly, and random. Seeds and random states can be found in the notebooks in the model folder.

Preprocessing
- Imputation: Simple imputation (using the mean) was performed on BMI, since only a small proportion of values (4%) were missing from the dataset.
- About 30% of the smoking status variable was labeled was "Unknown". We experimented with random imputation but decided to leave smoking status as is because imputing such a large proportion of data would lead to bias and might not represent the true data distribution. Thus, the interpretation of smoking status is a limitation of this analysis.
- There was 1 observation with "Other" gender, which was grouped with Male gender to reduce noise.
- Observations with "children" as the work type was grouped with the "never worked" category.
- Standard Scaling was done in the logistic regression model for the numerical variables age, bmi, and average glucose level
- Categorical variables were one-hot encoded.

Techniques & Metrics
- The dataset was split using train_test_split from sklearn.model_selection.
- Grid search was used for hyperparameter tuning of the models.
- Performance metrics include accuracy, precision, recall, F1, and ROC AUC (Receiver Operating Characteristic - Area Under the Curve). 
- The features of the models were examined with feature importance and SHAP value plots.

Models
- Since this is a classification problem, the processed data was first fit into a logistic regression model with balanced class weights to adjust for class imbalance. VIF values were checked and no multi-collinearity was detected. All of the non-outcome variables were fit into the model. Tuning of hyperparameters was also done using sklearn's GridSearchCV to optimize for recall, metrics from the best model are shown in Key Findings.
- Since we have significantly more non-stroke observations than stroke observations in our data, we also implemented XGBoost (Extreme Gradient Boosting) to address the class imbalance. XGBoost allows us to adjust the weight of the minority class to compensate for the imbalance, pushing the model to focus on predicting strokes.

## Key Findings & Instructions
Setup instructions include loading the libraries in Python and performing the preprocessing steps (see preprocessing.ipynb in data/processed).

If we optimize for recall in an XGBoost, for our 1022 test observations:

972 patients did not have a stroke
  - 407 patients who did not have a stroke were correctly classified as NOT having a stroke
  - 565 patients who did not have a stroke were INCORRECTLY classified as having a stroke

50 patients did have a stroke
  - 48 patients who had a stroke were correctly classified as having a stroke
  - 2 patients who had a stroke were INCORRECTLY classified as NOT having a stroke

Since stroke is a serious adverse health event, our model may add value in that it is much worse to be incorrectly classified as not having a stroke when in fact there is a stroke (false negative), than misclassified as stroke when in fact there is no stroke.

| Metric (Optimized for Recall) | Logistic Regression | XGBoost Model |
|----------|----------|----------|
| Accuracy | 0.6624 | 0.4452 |
| Precision | 0.1108 | 0.0783 |
| F1 Score | 0.1958 | 0.1448 |
| Recall | 0.84 | 0.96 |
| ROC AUC | 0.8336 | 0.8389 |

Based on SHAP summaries of the tuned and untuned logistic regression models, the most important features were age, glucose level, work type, hypertension,BMI, and smoking status.

**SHAP Bee Swarm for Logistic Regression (recall)**

![alt text](images/shap_logreg_tuned.png)

Based on SHAP summaries of the XGBoost model, the most important features were age, BMI, work type = self-employed, hypertension, glucose level.

**SHAP Bee Swarm for XGBoost (recall)**

![alt text](images/xgboost-recall-SHAP.png)

In conclusion, our recommendation for which factors significantly predict the occurence of stroke is mainly "Age". "BMI" and "average glucose level" are also noteworthy.

![alt text](images/age_box_whiskers.png)

One actionable insight would be to recommend screening for stroke risk at certain age thresholds.

![alt text](images/glucose.png)

Stroke cases tend to have higher glucose levels compared to non-stroke cases.
There is a second peak around 200–250 mg/dL, suggesting hyperglycemia (high blood sugar) is more common in stroke patients. The distribution is more spread out, indicating greater variability in glucose levels among stroke patients.

![alt text](images/BMI.png)

Both distributions peak around 25–30 BMI, which is in the overweight range. But the overall shapes of the curves are quite similar.

**Risks, Unknowns, & Limitations**

Limitation: Due to the rare outcome of stroke in this dataset, it was challenging to optimize on several performance metrics. Our final models showed good recall and AUC scores (over 0.8 in the logistic regression model and over 0.9 in the XGBoost model). Due to the various health conditions among individuals for a stroke to occure, it might not be possible to predict the timing of a stroke. However, we were able to find several key factors that would put one at the risk of a stroke, with age being the most important. 

Unknown: The source of the dataset is unknown, and might not be representative of the distribution in the population. Although, the 4% stroke outcome seen in this dataset is similar to the 3% prevalence in Canadian adults, according to the [Public Health Agency of Canada](https://health-infobase.canada.ca/ccdss/data-tool/). 

Unknown: The dataset lacks information on other health conditions and lifestyle factors which have been found to be highly associated with strokes, including prior stroke history, transient ischemic attack history, blood cholesterol, alcohol consumption, etc. [(source)](https://www.hopkinsmedicine.org/health/conditions-and-diseases/stroke)

Unknown: The dataset is cross-sectional. We only chose stroke as the outcome because that's our business problem. However, the outcome could be other variables such as hypertension and heart disease. Thus we cannot determine causal relationship between stroke and the other variables, only associations.

Risk: Depending on the prevention effort on individuals who are incorrectly predicted to have a stroke according to our model, the risk of implementing our modelling results would be the balance between harm and benefit in the prevention. For example, if the prevention strategy had negative side effects, or could be dangerous and invasive, the model results should not be used because of its low precision (which was the cost of high recall). Since our business case is focused on insights for public health decision makers, the prevention strategy is likely to be one implemented at the upstream level such as patient education efforts (e.g. smoking cessation, diet and exercise), increasing primary care, and managing other health conditions.

Limitation: As mentioned above, we experimented with imputing the "Unknown" smoking status category, but imputing such a large proportion of data would lead to bias and might not represent the true data distribution. Smoking is a critical predictor of stroke, so unknown smoking status is a limitation in this dataset.

**Next Steps**

If our team had more time we would like to continue working on a model with high performance metrics. This would involve gathering more data, researching advance methods to address class imbalance, and resolving the "unknown" smoking statuses.

It could also be worthwhile experimenting modelling age as a non-linear relationship, or categorizing into age groups to find the one with the highest likelihood of stroke, which would help with a public health screening strategy.

We would also further investigate the relationships between heart disease and stroke as well as hypertension and stroke. It was interesting that heart disease did not rank high in our SHAP values despite existing evidence of heart disease increasing the risk of stroke.

## Visuals & Credits

Age & Stroke

![alt text](images/age-curves.png)

Scatterplots of Age vs Average Glucose Level and Age vs BMI 

![alt text](images/scatterplots.png)

Heart disease

![alt text](images/hd.png)

Hypertension

![alt text](images/ht.png)

#### Plots of categorical features
![alt text](images/categorical_col.png)

## Credits and links to personal videos
* [Rui Qian Sun](https://drive.google.com/file/d/1MoJs6cyk0mN9n4DkvcrG86FDFUr61uZS/view?usp=sharing)
* [Catherine Liang](https://drive.google.com/file/d/19oSmHOiZjex43_U9BETLY-BGkWk7g069/view?usp=sharing)
* [Mahbub Khandoker](https://drive.google.com/file/d/1MUOKuyR9tyO2AD_71izG7gaiRIc3NoQk/view?usp=drivesdk)
* [Neethila Poddar](https://drive.google.com/file/d/1ojXnrVzG1xlFsG_niDt0VCXGzpTrTWXR/view?usp=sharing)
* [Devangi Vyas](https://drive.google.com/file/d/1HRNICQayW3dthcMs71Ejw2DBqPIYJ66v/view?usp=sharing)
