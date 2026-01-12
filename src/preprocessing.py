def preprocess(df):
    df = df.rename(columns={
        'Total CHOL': 'Total_Cholesterol',
        'HDL chol': 'HDL',
        'TRIGLYCERIDES': 'Triglycerides',
        'DIRECT LDL': 'Direct_LDL'
    })

    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    df = df.dropna(subset=[
        'Gender','Age',
        'Total_Cholesterol','HDL',
        'Triglycerides','Direct_LDL'
    ])

    df = df[
        (df['Total_Cholesterol'] > 0) &
        (df['HDL'] > 0) &
        (df['Triglycerides'] > 0) &
        (df['Direct_LDL'] > 0) &
        (df['Age'] > 0)
    ]

    return df
