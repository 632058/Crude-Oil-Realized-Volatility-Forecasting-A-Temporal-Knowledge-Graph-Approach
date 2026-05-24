import os
import pickle
import duckdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# =========================
# Configuration
# =========================
DB_PATH = "data/gdelt_daily_bilateral_by_eventcode_add.duckdb"
SRC_TABLE = "daily_bilateral_eventbasecode_sorted"
OUT_TABLE = "training_ready_data"
MAPPING_FILE = "data/gdelt_mappings.pkl"

RV_CSV = "data/rv.csv"
GPR_XLS = "data/data_gpr_daily_recent.xls"
OUTPUT_RV_GPR_CSV = "data/rv_with_gpr.csv"


# =========================
# 1. Build GNN-ready database structure
# =========================
def build_gnn_training_data():
    """
    Build the GNN-ready training table and mapping file.

    This function:
    1. Reads the raw bilateral event table from DuckDB.
    2. Builds integer mappings for entities, relations, and dates.
    3. Converts the raw event table into model-ready integer IDs.
    4. Writes the processed result into `training_ready_data`.
    5. Saves the mapping objects to `gdelt_mappings.pkl`.
    """
    if os.path.exists(MAPPING_FILE):
        print("Mapping file already exists. Rebuilding will overwrite the existing mappings.")

    con = duckdb.connect(DB_PATH)
    print("Step 1: Building ID mappings...")

    # Read all unique entities, relations, and dates
    entities = con.execute(
        f"""
        SELECT DISTINCT src_country AS entity FROM {SRC_TABLE}
        UNION
        SELECT DISTINCT dst_country AS entity FROM {SRC_TABLE}
        """
    ).df()["entity"].tolist()

    relations = con.execute(
        f"SELECT DISTINCT event_code FROM {SRC_TABLE}"
    ).df()["event_code"].tolist()

    dates = con.execute(
        f"SELECT DISTINCT date_added FROM {SRC_TABLE} ORDER BY date_added"
    ).df()["date_added"].tolist()

    entity_le = LabelEncoder().fit(entities)
    rel_le = LabelEncoder().fit(relations)
    date_map = {date: idx for idx, date in enumerate(dates)}

    mappings = {
        "entity_le": entity_le,
        "rel_le": rel_le,
        "date_map": date_map,
        "idx_to_date": {v: k for k, v in date_map.items()},
        "num_entities": len(entities),
        "num_relations": len(relations),
    }

    with open(MAPPING_FILE, "wb") as f:
        pickle.dump(mappings, f)

    print(f"Saved mapping file to: {MAPPING_FILE}")
    print(f"Step 2: Transforming raw event data and writing table `{OUT_TABLE}`...")

    df = con.execute(f"SELECT * FROM {SRC_TABLE}").df()

    # Transform raw categorical columns into integer IDs
    df["src_id"] = entity_le.transform(df["src_country"])
    df["dst_id"] = entity_le.transform(df["dst_country"])
    df["rel_id"] = rel_le.transform(df["event_code"])
    df["time_idx"] = df["date_added"].map(date_map)

    # Log-transform event counts for model training
    df["log_weight"] = np.log1p(df["event_count"])

    # Recreate the output table
    con.execute(f"DROP TABLE IF EXISTS {OUT_TABLE}")

    model_df = df[["time_idx", "src_id", "dst_id", "rel_id", "log_weight"]].copy()
    con.register("temp_df_for_insert", model_df)
    con.execute(f"CREATE TABLE {OUT_TABLE} AS SELECT * FROM temp_df_for_insert")
    con.unregister("temp_df_for_insert")

    # Create index for faster daily lookup
    con.execute(f"CREATE INDEX IF NOT EXISTS idx_time ON {OUT_TABLE} (time_idx)")

    print("Running VACUUM to reclaim space and optimize the database file...")
    con.execute("VACUUM")

    con.close()
    print(f"✓ GNN-ready table `{OUT_TABLE}` has been created successfully.")
    print("✓ The database now includes the `log_weight` field for model input.")


# =========================
# 2. Build rv_with_gpr.csv
# =========================
def build_rv_with_gpr():
    """
    Merge realized volatility data with daily GPR indicators.

    This function:
    1. Reads the original rv.csv file.
    2. Reads the daily GPR Excel file.
    3. Reshapes the GPR data from wide format to long format.
    4. Aligns the columns with rv.csv.
    5. Appends the GPR rows to the original RV dataset.
    6. Saves the merged result as rv_with_gpr.csv.
    """
    print("Step 3: Building merged RV + GPR dataset...")

    rv = pd.read_csv(RV_CSV)
    rv["date"] = pd.to_datetime(rv["date"])

    df_gpr = pd.read_excel(GPR_XLS)
    df_gpr["date"] = pd.to_datetime(df_gpr["date"])

    # Convert wide-format GPR columns into long-format symbol rows
    gpr_melted = pd.melt(
        df_gpr,
        id_vars=["date"],
        value_vars=["GPRD", "GPRD_ACT", "GPRD_THREAT"],
        var_name="symbol",
        value_name="value"
    )

    # Assign the same value to both close and rv
    gpr_melted["close"] = gpr_melted["value"]
    gpr_melted["rv"] = gpr_melted["value"]
    gpr_melted = gpr_melted.drop(columns=["value"])

    # Add missing columns so that the schema matches rv.csv
    missing_cols = [col for col in rv.columns if col not in gpr_melted.columns]
    for col in missing_cols:
        gpr_melted[col] = np.nan

    # Reorder columns to match rv.csv exactly
    gpr_melted = gpr_melted[rv.columns]

    # Append the GPR rows to the original RV dataset
    combined_df = pd.concat([rv, gpr_melted], ignore_index=True)

    # Sort for readability and downstream consistency
    combined_df = combined_df.sort_values(by=["date", "symbol"]).reset_index(drop=True)

    # Format date string consistently
    combined_df["date"] = pd.to_datetime(combined_df["date"]).dt.strftime("%Y-%m-%d 00:00:00.000")

    combined_df.to_csv(OUTPUT_RV_GPR_CSV, index=False)
    print(f"✓ Merged dataset saved to: {OUTPUT_RV_GPR_CSV}")


# =========================
# 3. Optional inspection utilities
# =========================
def inspect_output_summary():
    """
    Print a simple summary of the generated outputs.
    """
    print("\n" + "=" * 60)
    print("Output summary")
    print("=" * 60)

    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, "rb") as f:
            mappings = pickle.load(f)
        print(f"Mapping file: {MAPPING_FILE}")
        print(f"  Number of entities : {mappings['num_entities']}")
        print(f"  Number of relations: {mappings['num_relations']}")
        print(f"  Number of dates    : {len(mappings['date_map'])}")

    if os.path.exists(OUTPUT_RV_GPR_CSV):
        df = pd.read_csv(OUTPUT_RV_GPR_CSV)
        print(f"\nMerged CSV: {OUTPUT_RV_GPR_CSV}")
        print(f"  Rows    : {len(df)}")
        print(f"  Columns : {len(df.columns)}")
        if "symbol" in df.columns:
            print(f"  Symbols : {sorted(df['symbol'].dropna().unique().tolist())[:10]} ...")


# =========================
# 4. Main execution logic
# =========================
def main():
    print("=" * 60)
    print("Data preparation pipeline")
    print("=" * 60)

    build_gnn_training_data()
    build_rv_with_gpr()
    inspect_output_summary()

    print("\n✓ Data preparation completed successfully.")


if __name__ == "__main__":
    main()