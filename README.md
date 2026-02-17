# Churn Intelligence Platform

A machine learning project that generates synthetic customer data, analyzes behavioral patterns, and predicts customer churn for a subscription-based business.


## Project Structure

```
churn-intelligence-platform/
├── data/                # Raw and processed data files (CSVs)
├── notebooks/           # Jupyter notebooks for EDA and experimentation
├── src/                 # Source code (generators, feature engineering, utilities)
├── models/              # Trained and serialized ML models
├── config/              # Configuration files (schema definitions, hyperparameters)
├── requirements.txt     # Python package dependencies
└── README.md
```


## Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/churn-intelligence-platform.git
cd churn-intelligence-platform

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
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


## Data Generation Notes

- Customer ages follow a realistic distribution (weighted toward 25-45, fewer 60+)
- Signup dates are spread across the last 2 years
- Monthly activity starts with high engagement for new customers
- Some customers follow a decay pattern (gradual drop in logins, sessions, feature usage)
- Seasonality is applied: usage tends to drop in summer months and spike in January
- Churn correlates with declining engagement, increased support tickets, and payment failures


## Tech Stack

- Python 3.12
- pandas, numpy -- data manipulation
- scikit-learn -- machine learning
- matplotlib, seaborn -- visualization
- faker -- synthetic data generation
