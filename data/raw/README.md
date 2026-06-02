# Data — Home Credit Default Risk

This directory holds the raw Home Credit Default Risk dataset. **The data files are not committed to the repository** (they exceed reasonable repo size, and Kaggle's competition rules restrict redistribution).

## How to download

### 1. Install the Kaggle CLI

```bash
pip install kaggle
```

### 2. Set up API credentials

- Go to <https://www.kaggle.com/settings/account>
- Click **Create New Token** under the API section. This downloads `kaggle.json`.
- Move it to the expected location:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Accept the competition rules

You must accept the competition rules on Kaggle before downloading: <https://www.kaggle.com/competitions/home-credit-default-risk/rules>

### 4. Download into this directory

From the repo root:

```bash
cd data/raw
kaggle competitions download -c home-credit-default-risk
unzip home-credit-default-risk.zip
rm home-credit-default-risk.zip
```

## What you need for v1

This project uses only the two main applicant tables:

- `application_train.csv` (~286 MB, 307,511 rows × 122 columns) — training data with `TARGET`
- `application_test.csv` (~46 MB, 48,744 rows × 121 columns) — held-out applicants without `TARGET`

The auxiliary tables (`bureau.csv`, `previous_application.csv`, `credit_card_balance.csv`, `installments_payments.csv`, `POS_CASH_balance.csv`, `bureau_balance.csv`) are **not used in v1** — see the main README for the scoping rationale.

You can delete the auxiliary tables after extraction if disk space matters.
