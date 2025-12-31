import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

    # POINT 3: Class 
class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    # POINT 4: Loading Data method
    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        print("✅ Data successfully loaded!")

    # POINT 5: Handle missing values
    def handle_missing_values(self):
        """Missing values ko fill karne ka method"""
        # Age ko median se fill karna
        self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median())
        # Embarked ko mode (most frequent) se fill karna
        self.df['Embarked'] = self.df['Embarked'].fillna(self.df['Embarked'].mode()[0])
        print("✅ Missing values handled!")

    # POINT 6: Encode categorical using LabelEncoder
    def encode_categorical(self):
        """Categorical data (Sex, Embarked) ko numbers mein badalne ka method"""
        le = LabelEncoder()
        self.df['Sex'] = le.fit_transform(self.df['Sex'])
        self.df['Embarked'] = le.fit_transform(self.df['Embarked'])
        print("✅ Categorical columns encoded!")

    # POINT 7: Implement Scale Features method
    def scale_features(self):
        """Numerical features ko normalize/scale karne ka method"""
        scaler = StandardScaler()
        # Sirf Age aur Fare ko scale karte hain
        self.df[['Age', 'Fare']] = scaler.fit_transform(self.df[['Age', 'Fare']])
        print("✅ Features scaled successfully!")

    # POINT 8: Add Split Data() method
    def split_data(self, target_column):
        """Data ko Train aur Test sets mein divide karna"""
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"✅ Data split into Train ({len(X_train)}) and Test ({len(X_test)}) sets!")
        return X_train, X_test, y_train, y_test
    
    
    # POINT 9: Create save_processed_data() method
    def save_processed_data(self, output_path):
        """Cleaned aur processed data ko CSV file mein save karna"""
        if self.df is not None:
            self.df.to_csv(output_path, index=False)
            print(f"✅ Processed data saved to: {output_path}")
            

# POINT 10: Demonstrate usage in main block
if __name__ == "__main__":
    # Input file ka path (Jo pehle se cleaned data tha)
    path = r'week2\TAsk_2_2_Pandas\titanic_cleaned.csv'
    
    # Output file ka path (Isi folder mein save karne ke liye sirf naam likhen)
    output_filename = 'final_processed_data.csv'
    
    preprocessor = DataPreprocessor(path)
    
    # Usage Demonstration
    preprocessor.load_data()
    preprocessor.handle_missing_values()
    preprocessor.encode_categorical()
    preprocessor.scale_features()
    
    # Task 2.4 ke folder mein hi save hoga
    preprocessor.save_processed_data('week2/Task_2_4_OOP/final_processed_data.csv')
    
    # Splitting for ML
    X_train, X_test, y_train, y_test = preprocessor.split_data('Survived')