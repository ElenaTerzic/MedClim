import pandas as pd

# Read your full Argo file
df = pd.read_csv('/Users/elena/Projects/IRB/MedClim/Data/Argo/Merged/argo_med_all_sorted.csv')

# Make sure QC columns are strings (covers both int and char cases)
for col in ['PRES_QC', 'TEMP_QC', 'PSAL_QC']:
    df[col] = df[col].astype(str)

# Define what a "profile" is: WMO (PLATFORM_NUMBER) + CYCLE_NUMBER
profile_cols = ['PLATFORM_NUMBER', 'CYCLE_NUMBER']

# Build a per-profile summary table
profile_qc = (
    df.groupby(profile_cols)
      .agg({
          # If any level in a profile has QC=2, mark the whole profile as 2, else 1
          'PRES_QC': lambda x: '2' if (x == '2').any() else '1',
          'TEMP_QC': lambda x: '2' if (x == '2').any() else '1',
          'PSAL_QC': lambda x: '2' if (x == '2').any() else '1',
          # Take first value for geo/time (they should be constant within a profile)
          'LATITUDE': 'first',
          'LONGITUDE': 'first',
          'TIME': 'first',
      })
      .reset_index()
)

# Total number of profiles
total_profiles = len(profile_qc)

# Per-variable profile QC stats
pres_qc1_profiles = (profile_qc['PRES_QC'] == '1').sum()
pres_qc2_profiles = (profile_qc['PRES_QC'] == '2').sum()

temp_qc1_profiles = (profile_qc['TEMP_QC'] == '1').sum()
temp_qc2_profiles = (profile_qc['TEMP_QC'] == '2').sum()

psal_qc1_profiles = (profile_qc['PSAL_QC'] == '1').sum()
psal_qc2_profiles = (profile_qc['PSAL_QC'] == '2').sum()

# Overall: profiles with any QC=2 in any variable
mask_any_qc2 = (profile_qc[['PRES_QC', 'TEMP_QC', 'PSAL_QC']] == '2').any(axis=1)
overall_qc2_profiles = mask_any_qc2.sum()
overall_qc1_profiles = total_profiles - overall_qc2_profiles

# Print summary
print(f"Total profiles: {total_profiles}\n")

print("Pressure (per profile):")
print(f"  QC=1: {pres_qc1_profiles} ({pres_qc1_profiles/total_profiles*100:.1f}%)")
print(f"  QC=2: {pres_qc2_profiles} ({pres_qc2_profiles/total_profiles*100:.1f}%)\n")

print("Temperature (per profile):")
print(f"  QC=1: {temp_qc1_profiles} ({temp_qc1_profiles/total_profiles*100:.1f}%)")
print(f"  QC=2: {temp_qc2_profiles} ({temp_qc2_profiles/total_profiles*100:.1f}%)\n")

print("Salinity (per profile):")
print(f"  QC=1: {psal_qc1_profiles} ({psal_qc1_profiles/total_profiles*100:.1f}%)")
print(f"  QC=2: {psal_qc2_profiles} ({psal_qc2_profiles/total_profiles*100:.1f}%)\n")

print("Overall profiles (any variable with QC=2):")
print(f"  All-QC1 profiles: {overall_qc1_profiles} ({overall_qc1_profiles/total_profiles*100:.1f}%)")
print(f"  Any-QC2 profiles: {overall_qc2_profiles} ({overall_qc2_profiles/total_profiles*100:.1f}%)")

# ---- CSV with only profiles that have QC=2 somewhere ----

# Filter only profiles where at least one variable has QC=2
qc2_profiles = profile_qc.loc[mask_any_qc2].copy()

def vars_qc2(row):
    vars_ = []
    if row['PRES_QC'] == '2':
        vars_.append('pressure')
    if row['TEMP_QC'] == '2':
        vars_.append('temperature')
    if row['PSAL_QC'] == '2':
        vars_.append('salinity')
    return ";".join(vars_)

qc2_profiles['vars_qc2'] = qc2_profiles.apply(vars_qc2, axis=1)

# Save only requested columns
qc2_profiles[['PLATFORM_NUMBER', 'LATITUDE', 'LONGITUDE', 'TIME', 'vars_qc2']].to_csv(
    '/Users/elena/Projects/IRB/MedClim/Data/Argo/CSV/argo_qc2_profiles.csv',
    index=False
)
