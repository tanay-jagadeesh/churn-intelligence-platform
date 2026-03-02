# Churn Intelligence Platform

A machine learning pipeline that predicts customer churn for a subscription-based business, segments customers by risk level, and estimates the revenue impact of retention efforts.


## What It Does

1. **Generates synthetic customer data** -- 2,000 customer profiles and ~24,000 monthly activity records with realistic behavioral patterns
2. **Engineers predictive features** -- transforms raw monthly time-series into per-customer summary statistics (averages, trends, payment reliability)
3. **Trains a Random Forest classifier** -- predicts churn with ~98% accuracy and reports which features matter most
4. **Segments customers by risk** -- assigns every customer a High / Medium / Low risk tier with recommended actions
5. **Calculates retention ROI** -- estimates how much revenue is at risk and how much can be saved through intervention


## Project Structure

```
churn-intelligence-platform/
├── config/
│   └── schema.py            # All constants and hyperparameters
├── src/
│   ├── generators.py        # Synthetic data generation (customers + activity)
│   ├── features.py          # Feature engineering (raw data → model input)
│   ├── model.py             # Model training, evaluation, risk segmentation, ROI
│   └── utils.py             # Save/load helpers for CSVs and models
├── notebooks/
│   └── 01_eda.ipynb         # Exploratory data analysis with visualizations
├── data/                    # Generated CSVs (customers, activity, features, risk segments)
├── models/                  # Serialized trained model (churn_model.joblib)
├── main.py                  # CLI entry point — runs the full pipeline
├── requirements.txt
└── README.md
```


## Setup

```bash
git clone https://github.com/<your-username>/churn-intelligence-platform.git
cd churn-intelligence-platform

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```


## Usage

```bash
# Run the full pipeline (generate data → train → evaluate → segment → ROI)
python main.py

# Only generate data
python main.py --generate

# Only train and evaluate (data must already exist in data/)
python main.py --train
```


## Sample Output

```
Generating customer data...
  Saved 2000 customers to data/customers.csv
  Churn rate: 469/2000 (23.4%)
Generating monthly activity data
  Saved 24354 activity records to data/monthly_activity.csv

Accuracy: 98.8%

Top 10 Most Important Features
                feature  importance
recent_avg_satisfaction    0.1611
       avg_satisfaction    0.1357
            login_trend    0.1215
      recent_avg_logins    0.1182
    recent_avg_sessions    0.0951

Risk Segmentation
  High  :  438 customers (21.9%)
  Medium:   81 customers (4.0%)
  Low   : 1481 customers (74.1%)

Retention ROI Analysis
  Total revenue at risk:  $698,030.28/year
  Projected savings:      $ 59,905.96/year
```


## Data Schema

### Customer Profile (customers.csv)

Static attributes collected at signup or rarely changed.

| Field              | Type     | Description                                      |
|--------------------|----------|--------------------------------------------------|
| customer_id        | string   | Unique identifier for each customer              |
| name               | string   | Customer full name                               |
| email              | string   | Customer email address                           |
| age                | int      | Customer age (distribution weighted toward 25-45)|
| gender             | string   | Male / Female / Other                            |
| location           | string   | City or region                                   |
| signup_date        | date     | Account creation date (spread over last 2 years) |
| plan_type          | string   | Basic / Standard / Premium                       |
| monthly_charge     | float    | Monthly subscription cost in USD                 |
| acquisition_channel| string   | How the customer found us (referral, ad, organic, social media) |
| contract_type      | string   | Month-to-month / Annual                          |
| churned            | bool     | Whether the customer has canceled (target label)  |
| churn_date         | date     | Date of cancellation (null if still active)      |

### Monthly Activity (monthly_activity.csv)

Behavioral data tracked each month for every active customer.

| Field              | Type     | Description                                      |
|--------------------|----------|--------------------------------------------------|
| customer_id        | string   | Links to customer profile                        |
| month              | date     | The month this record covers (YYYY-MM)           |
| logins             | int      | Number of times the customer logged in            |
| feature_usage_score| float    | Composite score (0-100) of how many features used|
| total_sessions     | int      | Total app/site sessions in the month             |
| avg_session_duration| float   | Average session length in minutes                |
| support_tickets    | int      | Number of support tickets opened                 |
| payment_status     | string   | On-time / Late / Failed                          |
| satisfaction_score | float    | Monthly NPS or satisfaction rating (1-10)        |

### Engineered Features (features.csv)

One row per customer with computed summary statistics used as model input.

| Feature                 | Description                                          |
|-------------------------|------------------------------------------------------|
| avg_logins              | Mean monthly logins across entire history             |
| recent_avg_logins       | Mean monthly logins over the last 3 months            |
| login_trend             | Slope of logins over time (negative = declining)      |
| avg_sessions            | Mean monthly sessions                                |
| recent_avg_sessions     | Mean sessions over last 3 months                     |
| session_trend           | Slope of sessions over time                          |
| avg_satisfaction        | Mean satisfaction score                              |
| recent_avg_satisfaction | Mean satisfaction over last 3 months                 |
| satisfaction_trend      | Slope of satisfaction over time                      |
| avg_feature_usage       | Mean feature usage score                             |
| avg_duration            | Mean session duration in minutes                     |
| total_support_tickets   | Total support tickets filed                          |
| tenure_months           | Number of months as a customer                       |
| ontime_payment_rate     | Fraction of payments that were on-time (0 to 1)      |


## Data Generation Notes

- Customer ages follow a normal distribution (mean=35, std=10, clipped to 18-72)
- Signup dates are spread across the last 2 years
- Monthly activity starts with high engagement for new customers
- Churners follow a linear decay pattern (engagement drops from 100% to 20% over their lifetime)
- Seasonality is applied: usage drops in summer months (Jun-Aug) and spikes in January
- Churn correlates with declining engagement, increased support tickets, and payment failures
- Annual contract customers have a lower churn probability than month-to-month


## Risk Segmentation

Customers are scored with a churn probability (0-100%) and bucketed into tiers:

| Tier   | Probability | Recommended Action                                    |
|--------|-------------|-------------------------------------------------------|
| High   | > 70%       | Offer discount or personal outreach from account manager |
| Medium | 30-70%      | Send re-engagement email campaign with feature highlights |
| Low    | < 30%       | No action needed -- continue monitoring                |


## Tech Stack

- Python 3.12
- pandas, numpy -- data manipulation
- scikit-learn -- machine learning (Random Forest)
- matplotlib, seaborn -- visualization
- faker -- synthetic data generation
- joblib -- model serialization
