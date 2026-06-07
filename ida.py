import os
import pandas as pd

# =====================================================================
# 1. LOAD THE DATASET FILE
# =====================================================================
TRAIN_FILE = "data.csv"

# Verify that the data file exists in the directory before starting
if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"❌ Critique Error: The file `{TRAIN_FILE}` was not found in the current directory.\n"
        f"Please ensure it is saved alongside this script before execution."
    )

# Read the file and strip any accidental whitespace from the column headers
df = pd.read_csv(TRAIN_FILE, sep=";")
df.columns = df.columns.str.strip()

print("==============================================================")
print("             INITIAL DATA ANALYSIS (IDA) REPORT               ")
print("==============================================================\n")

# =====================================================================
# 2. STRUCTURAL METADATA
# =====================================================================
print("## 1. Structural Metadata & Dimensions")
print(f"🔹 Total Number of Observations (Rows): {df.shape[0]}")
print(f"🔹 Total Number of Variables (Columns): {df.shape[1]}")
print("\n📊 Column Data Types:")
print(df.dtypes)
print("-" * 62)

# =====================================================================
# 3. DATA QUALITY & COMPLETENESS (Missing Values)
# =====================================================================
print("\n## 2. Completeness & Missing Values Check")
missing_count = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing Values': missing_count,
    'Percentage (%)': [f"{p:.2f}%" for p in missing_percentage]
})
print(missing_df)

if df.isnull().sum().sum() == 0:
    print("\n✅ Perfect Integrity: Zero missing values detected across all fields.")
else:
    print("\n⚠️ Warning: Missing elements found. Handle gaps before model insertion.")
print("-" * 62)

# =====================================================================
# 4. RANGE & DOMAIN VALIDATION (Sanity Check)
# =====================================================================
print("\n## 3. Boundary & Range Validation")
print("Checking for illegal values (e.g., percentages or grades outside [0, 100]):")

# Define columns expected to strictly follow a 0-100 threshold
bounded_100_cols = [
    'attendance_pct', 'homework_pct', 'midterm_score', 'informatique', 
    'mathematique', 'svt', 'physique', 'sport', 'education_islamique', 
    'francais', 'arabe'
]

anomalies = 0
for col in bounded_100_cols:
    if col in df.columns:
        out_of_bounds = df[(df[col] < 0) | (df[col] > 100)]
        if not out_of_bounds.empty:
            print(f"❌ Structural Anomaly in `{col}`: Found values outside 0-100 threshold!")
            print(out_of_bounds[['student_id', col]])
            anomalies += 1

if anomalies == 0:
    print("✅ Scale Integrity Verified: All grades and percentages reside properly within [0, 100].")
print("-" * 62)

# =====================================================================
# 5. TARGET CLASS BALANCE
# =====================================================================
print("\n## 4. Target Class Balance (`pass` Variable)")
# =====================================================================
# 5. TARGET CLASS BALANCE
# =====================================================================
print("\n## 4. Target Class Balance (`pass` Variable)")
if 'pass' in df.columns:
    # Clean the pass column using your model's exact cleaning regex logic
    # This ensures strings like "1" or "0" are cast properly to integers
    y_clean = df['pass'].astype(str).str.extract(r'(\d)')[0].astype(int)
    
    class_counts = y_clean.value_counts()
    class_percentages = y_clean.value_counts(normalize=True) * 100

    for class_label in [1, 0]:
        count = class_counts.get(class_label, 0)
        pct = class_percentages.get(class_label, 0.0)
        # FIXED: Correctly assigns PASS to 1 and FAIL to 0
        role = "PASS (1)" if class_label == 1 else "FAIL (0)" 
        print(f"🔹 Class {role} : {count} instances ({pct:.2f}%)")
else:
    print("ℹ️ Info: No `pass` column present in this dataset (Testing/Unlabeled Mode).")