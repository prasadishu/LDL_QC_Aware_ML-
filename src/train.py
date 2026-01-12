from sklearn.model_selection import train_test_split

def split_data(df):
    X = df[['Total_Cholesterol','HDL','Triglycerides','Age','Gender']]
    y = df['Direct_LDL']
    return train_test_split(X, y, test_size=0.2, random_state=42)
