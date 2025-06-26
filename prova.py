import pandas as pd
df = pd.read_csv("uploads/study_population_with_intervention.csv")
subset = df[df['X'] == 1]
subset['Y'].value_counts(normalize=True)
df_trans = pd.read_csv("uploads/transport_result_X_1.csv")
print(df_trans)
