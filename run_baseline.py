import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# =========================
# Parameter settings
# =========================
OIL_CSV = "data/rv_with_gpr.csv"
SYMBOL = "CL_c1"
OUTPUT_DIR = "baseline_results"

ROLLING_WINDOW_SIZE = 1000

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")


# =========================
# 1. Data preparation
# =========================
def prepare_har_data(csv_path, symbol, horizons_map):
    print("Preparing HAR baseline data (including OVX and GPR)...")
    data = pd.read_csv(csv_path)

    df = data[data["symbol"] == symbol].sort_values(by="date", ascending=True).copy()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    ovx_df = data[data["symbol"] == "OVX"].copy()
    if not ovx_df.empty:
        ovx_df["date"] = pd.to_datetime(ovx_df["date"])
        ovx_df = ovx_df.set_index("date")[["rv"]].rename(columns={"rv": "OVX"})
        df = df.join(ovx_df, how="left")
        df["OVX"] = df["OVX"].ffill()
        df["log_OVX_d"] = np.log(df["OVX"] + 1e-10)
    else:
        print("Warning: symbol='OVX' not found.")
        df["log_OVX_d"] = 0.0


    # GPR: already at daily frequency in the CSV; keep only one log_GPRD column
    gpr_df = data[data["symbol"] == "GPRD"].copy()
    if not gpr_df.empty:
        gpr_df["date"] = pd.to_datetime(gpr_df["date"])
        gpr_df = gpr_df.set_index("date")[["rv"]].rename(columns={"rv": "GPRD"})
        df = df.join(gpr_df, how="left")
        n_missing = df["GPRD"].isna().sum()
        if n_missing > 0:
            print(f"  [GPRD] {n_missing} missing values filled by forward fill.")
        df["GPRD"] = df["GPRD"].ffill()
        df["log_GPRD"] = np.log(df["GPRD"] + 1e-10)
    else:
        print("Warning: symbol='GPRD' not found. HAR-OVX-GPR will be affected.")
        df["log_GPRD"] = 0.0

    # Standard HAR features
    df["RV_d"] = df["rv"]
    df["RV_w"] = df["rv"].rolling(window=5).mean()
    df["RV_m"] = df["rv"].rolling(window=22).mean()
    df["log_RV_d"] = np.log(df["RV_d"] + 1e-10)
    df["log_RV_w"] = np.log(df["RV_w"] + 1e-10)
    df["log_RV_m"] = np.log(df["RV_m"] + 1e-10)

    # HAR-J
    df["Jump"] = np.maximum(df["RV_d"] - df["RV_w"], 0)
    df["log_Jump"] = np.log(df["Jump"] + 1e-10)

    # Targets
    next_day_rv = df["rv"].shift(-1)
    for name, days in horizons_map.items():
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=days)
        df[name] = next_day_rv.rolling(window=indexer).mean()

    # Important change 1:
    # Drop NA only based on training features, while preserving the final rows
    # whose targets may still be NaN
    features_to_check = [
        "log_RV_d", "log_RV_w", "log_RV_m",
        "log_OVX_d",
        "log_GPRD", "log_Jump"
    ]
    df.dropna(subset=features_to_check, inplace=True)

    print(f"  Data range: {df.index.min().date()} → {df.index.max().date()}  (N={len(df)})")
    return df


# =========================
# 2. Evaluation metrics
# =========================
def qlike_loss_np(y_true, y_pred):
    eps = 1e-6
    y_pred = np.clip(y_pred, eps, None)
    y_true = np.clip(y_true, eps, None)
    ratio = y_true / y_pred
    return np.mean(ratio - np.log(ratio) - 1)


def calculate_r2_oos(y_true, y_pred, y_bench):
    mse_model = np.mean((y_true - y_pred) ** 2)
    mse_bench = np.mean((y_true - y_bench) ** 2)
    return 1 - (mse_model / (mse_bench + 1e-10))


# =========================
# 3. Rolling forecast
# =========================
def run_online_har_forecast_full_history(df, target_col, horizon_days,
                                         model_type="HAR", warmup_steps=252):
    base_cols = ["log_RV_d", "log_RV_w", "log_RV_m"]
    ovx_cols = ["log_OVX_d"]  

    if model_type == "HAR-J":
        feature_cols = base_cols + ["log_Jump"]
    elif model_type == "HAR-OVX":
        feature_cols = base_cols + ovx_cols
    elif model_type == "HAR-OVX-GPR":
        feature_cols = base_cols + ovx_cols + ["log_GPRD"]
    else:
        feature_cols = base_cols

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: {model_type} is missing columns {missing_cols}. Skipping.")
        return None, None

    scaler = StandardScaler()
    y_true_list, y_pred_list, y_bench_list, dates_list = [], [], [], []

    print(f"\nStarting {model_type} online training: {target_col}")
    start_index = max(warmup_steps + horizon_days, warmup_steps)
    progress_bar = tqdm(range(start_index, len(df)), desc=f"{model_type}-{target_col}")

    for i in progress_bar:
        row = df.iloc[i:i+1]

        valid_train_end = i - horizon_days
        train_start_idx = max(0, valid_train_end - ROLLING_WINDOW_SIZE)

        if (valid_train_end - train_start_idx) < warmup_steps:
            continue

        train_data = df.iloc[train_start_idx:valid_train_end].copy()

        # Important change 2:
        # Exclude rows whose target is still NaN when fitting the model
        train_data.dropna(subset=[target_col], inplace=True)

        if len(train_data) < warmup_steps:
            continue

        x_test_raw = row[feature_cols].values.reshape(1, -1)
        bench_val = train_data[target_col].mean()

        X_train_raw = train_data[feature_cols].values
        scaler.fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        y_train_log = np.log(train_data[target_col].values + 1e-10)

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train_log)

        x_test = scaler.transform(x_test_raw)
        pred_val = np.exp(model.predict(x_test)[0])

        # Store predictions even if target_col is NaN
        y_true_list.append(row[target_col].iloc[0])
        y_pred_list.append(pred_val)
        y_bench_list.append(bench_val)
        dates_list.append(df.index[i])

    res_df = pd.DataFrame({
        "date": dates_list,
        "Actual": y_true_list,
        "Pred": y_pred_list,
        "Bench": y_bench_list
    })

    if len(res_df) == 0:
        return None, res_df

    # Important change 3:
    # Remove rows with NaN Actual values before computing evaluation metrics
    valid_eval_df = res_df.dropna(subset=["Actual"])

    if len(valid_eval_df) < 2:
        return None, res_df

    y_true_eval = valid_eval_df["Actual"].values
    y_pred_eval = valid_eval_df["Pred"].values
    y_bench_eval = valid_eval_df["Bench"].values

    result_dict = {
        "Horizon": target_col,
        "Model": model_type,
        "R2_Standard": r2_score(y_true_eval, y_pred_eval),
        "R2_OOS": calculate_r2_oos(y_true_eval, y_pred_eval, y_bench_eval),
        "QLIKE": qlike_loss_np(y_true_eval, y_pred_eval),
        "MSE": mean_squared_error(y_true_eval, y_pred_eval),
        "Test_Size": len(y_true_eval),
        "Start_Date": str(valid_eval_df["date"].iloc[0].date()),
        "End_Date": str(valid_eval_df["date"].iloc[-1].date())
    }

    return result_dict, res_df


# =========================
# 4. Main execution logic
# =========================
if __name__ == "__main__":

    horizons_map = {
        "Target_1D": 1,
        "Target_1W": 5,
        "Target_2W": 10,
        "Target_3W": 15,
        "Target_1M": 22,
        "Target_2M": 44,
        "Target_3M": 66,
    }

    WARMUP_STEPS = 252
    df = prepare_har_data(OIL_CSV, SYMBOL, horizons_map)

    models = ["HAR", "HAR-J", "HAR-OVX", "HAR-OVX-GPR"]
    all_results = []

    print("\n" + "=" * 60)
    print(f"Starting HAR baseline run (output directory: {OUTPUT_DIR})")
    print("   Window Type : Fixed Rolling (1000)")
    print("   OVX Feature : log_OVX_d")
    print("   GPR Feature : log_GPRD only (daily raw values)")
    print("=" * 60)

    for target_name in horizons_map.keys():
        print(f"\n--- {target_name} ---")
        horizon_days = horizons_map[target_name]

        for model_type in models:
            res_dict, res_df = run_online_har_forecast_full_history(
                df, target_name, horizon_days, model_type, warmup_steps=WARMUP_STEPS
            )
            if res_dict is None:
                continue

            all_results.append(res_dict)
            print(f"{model_type} - {target_name}:")
            print(f"  R2_OOS : {res_dict['R2_OOS']:.6f}")
            print(f"  QLIKE  : {res_dict['QLIKE']:.6f}")

            pred_filename = os.path.join(
                OUTPUT_DIR, f"full_history_pred_{model_type}_{target_name}.csv"
            )
            res_df.to_csv(pred_filename, index=False)

    baseline_df = pd.DataFrame(all_results)

    print("\n" + "=" * 60)
    print("HAR Baseline Summary (Full History)")
    print("=" * 60)
    pd.options.display.float_format = "{:,.6f}".format
    if not baseline_df.empty:
        cols_to_show = ["Horizon", "Model", "R2_OOS", "QLIKE",
                        "Test_Size", "Start_Date", "End_Date"]
        print(baseline_df[cols_to_show])

    summary_filename = os.path.join(OUTPUT_DIR, "har_baseline_full_history_summary.csv")
    baseline_df.to_csv(summary_filename, index=False)
    print(f"\n✓ HAR baseline run completed. Results saved to {OUTPUT_DIR}/")