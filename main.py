import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def inspect_data(df):
    print(df.shape)
    print(df.columns)
    print(df.dtypes)
    df.info()
    print(df.isna().sum())
    print(df.duplicated().sum())
    print(df.describe())

def remove_duplicates(df):
    clean_df = df.drop_duplicates()
    return clean_df

def clean_invalid_ages(df):
    clean_df = df.copy()
    invalid_age_mask = (clean_df["Age"] < 0) | (clean_df["Age"] > 120)
    clean_df.loc[invalid_age_mask, "Age"] = float("nan")
    return clean_df


df = load_data("raw_customer_data.csv")
print(df.shape)
print(df.head(5))

inspect_data(df)

original_rows = len(df)
missing_before = df.isna().sum().sum()
duplicates_before = df.duplicated().sum()

print("Original Rows:", original_rows)
print("Missing Before:", missing_before)
print("Duplicates Before:", duplicates_before)

df = remove_duplicates(df)
print("Remaining Duplicates:", df.duplicated().sum())
print("Shape After Removing Duplicates:", df.shape)

invalid_age_mask = (df["Age"] < 0) | (df["Age"] > 120)
print("Invalid Ages Before:", invalid_age_mask.sum())
print(df.loc[invalid_age_mask, ["Customer_ID", "Age"]])

df = clean_invalid_ages(df)
print(df)

print("Missing Ages After:", df["Age"].isna().sum())
print("Invalid Ages After:", ((df["Age"] < 0) | (df["Age"] > 120)).sum())

def clean_invalid_charges(df):
    clean_df = df.copy()
    monthly_charges_mask = clean_df["Monthly_Charges"] < 0
    clean_df.loc[monthly_charges_mask, "Monthly_Charges"] = float("nan")
    total_charges_mask = clean_df["Total_Charges"] < 0
    clean_df.loc[total_charges_mask, "Total_Charges"] = float("nan")
    return clean_df

df = clean_invalid_charges(df)
print((df["Monthly_Charges"] < 0).sum())
print((df["Total_Charges"] < 0).sum())


def clean_text_columns(df):
    clean_df = df.copy()
    text_columns = ["Gender", "City", "Contract_Type", "Payment_Method"]
    for column in text_columns:
        clean_df[column] = clean_df[column].str.strip().str.lower()
    return clean_df


df = clean_text_columns(df)
print(df["City"].unique())
print(df["Contract_Type"].unique())



def fill_numeric_missing(df):
    clean_df = df.copy()
    numeric_columns = [
        "Age",
        "Monthly_Charges",
        "Total_Charges",
        "Support_Calls",
        "Internet_Usage"
    ]
    for column in numeric_columns:
        median_value = clean_df[column].median()
        clean_df[column] = clean_df[column].fillna(median_value)
    return clean_df


df = fill_numeric_missing(df)
print(df[[
    "Age",
    "Monthly_Charges",
    "Total_Charges",
    "Support_Calls",
    "Internet_Usage"
]].isna().sum())


def fill_categorical_missing(df):
    clean_df = df.copy()
    categorical_columns = ["Gender", "City"]
    for column in categorical_columns:
        clean_df[column] = clean_df[column].fillna("unknown")
    return clean_df

df = fill_categorical_missing(df)
print(df.isna().sum())


final_rows = len(df)
missing_after = df.isna().sum().sum()
removed_rows = original_rows - final_rows

print("Final Rows:", final_rows)
print("Missing Values After:", missing_after)
print("Removed Rows:", removed_rows)
print("Remaining Duplicates:", df.duplicated().sum())


def get_high_value_customers(df):
    high_value_mask = df["Total_Charges"] > 300000
    return df[high_value_mask].copy()

high_value_customers = get_high_value_customers(df)
print(high_value_customers.head(5))
print("High-value Customers:", len(high_value_customers))



def get_high_support_customers(df):
    high_support_mask = df["Support_Calls"] >= 5
    return df[high_support_mask].copy()

high_support_customers = get_high_support_customers(df)
print(high_support_customers.head(5))
print("High-support Customers:", len(high_support_customers))



def get_high_monthly_charge_customers(df):
    high_monthly_charge_mask = df["Monthly_Charges"] > 8000
    return df[high_monthly_charge_mask].copy()

high_monthly_charge_customers = get_high_monthly_charge_customers(df)
print(high_monthly_charge_customers.head(5))
print("High Monthly Charge Customers:", len(high_monthly_charge_customers))


def get_customers_by_churn(df, churn_status):
    churn_mask = df["Churn"] == churn_status
    return df[churn_mask].copy()

churned_customers = get_customers_by_churn(df, 1)
stayed_customers = get_customers_by_churn(df, 0)
print("Churned Customers:", len(churned_customers))
print("Stayed Customers:", len(stayed_customers))


def get_contract_churn_report(df):
    report = df.groupby("Contract_Type").agg(
        Total_Customers=("Customer_ID", "count"),
        Churned_Customers=("Churn", "sum"),
        Churn_Rate=("Churn", "mean")
    )

    report["Churn_Rate"] = report["Churn_Rate"] * 100
    return report

contract_churn_report = get_contract_churn_report(df)
print(contract_churn_report)


def get_city_churn_report(df):
    report = df.groupby("City").agg(
        Total_Customers=("Customer_ID", "count"),
        Churned_Customers=("Churn", "sum"),
        Churn_Rate=("Churn", "mean")
    )

    report["Churn_Rate"] = report["Churn_Rate"] * 100
    return report

city_churn_report = get_city_churn_report(df)
print(city_churn_report)


def get_average_charges_report(df):
    report = df.groupby("Contract_Type").agg(
        Average_Monthly_Charges=("Monthly_Charges", "mean"),
        Average_Total_Charges=("Total_Charges", "mean")
    )
    return report

average_charges_report = get_average_charges_report(df)
print(average_charges_report)


def get_support_calls_report(df):
    report = df.groupby("Churn").agg(
        Average_Support_Calls=("Support_Calls", "mean")
    )
    return report

support_calls_report = get_support_calls_report(df)
print(support_calls_report)



def prepare_features_target(df):
    X = df.drop(columns=["Customer_ID", "Churn"]).copy()
    y = df["Churn"].copy()
    return X, y

X, y = prepare_features_target(df)
print("X Shape:", X.shape)
print("y Shape:", y.shape)
print(X.columns)



def encode_features(X):
    X_encoded = pd.get_dummies(
        X,
        columns=["Gender", "City", "Contract_Type", "Payment_Method"],
        dtype=int
    )
    return X_encoded

X_encoded = encode_features(X)
print(X_encoded.head())
print(X_encoded.shape)


def export_datasets(df, X_encoded, y):
    df.to_csv("clean_customer_data.csv", index=False)
    ml_ready_data = X_encoded.copy()
    ml_ready_data["Churn"] = y
    ml_ready_data.to_csv("ml_ready_customer_data.csv", index=False)

export_datasets(df, X_encoded, y)


churn_rate = df["Churn"].mean() * 100
average_monthly_charges = df["Monthly_Charges"].mean()
highest_churn_segment = contract_churn_report["Churn_Rate"].idxmax()

print("Churn Rate (%):", churn_rate)
print("Average Monthly Charges:", average_monthly_charges)
print("Highest Churn Segment (Contract Type):", highest_churn_segment)


print("Total Records Before Cleaning:", original_rows)
print("Clean Customers:", final_rows)
print("Removed Duplicates:", duplicates_before)
print("Missing Values Before:", missing_before)
print("Missing Values After:", missing_after)
print("Churned Customers:", len(churned_customers))
print("Churn Rate (%):", churn_rate)
print("Average Monthly Charges:", average_monthly_charges)
print("Highest Churn Segment (Contract Type):", highest_churn_segment)






