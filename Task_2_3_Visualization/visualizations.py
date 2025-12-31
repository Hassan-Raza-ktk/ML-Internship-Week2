#Task 2.3 Data Visualization with Matplotlib and Seaborn by HASSAN RAZA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 2.3.1 Seaborn ALready installed Hai version 0.13.2 
print(sns.__version__)

# Step 2.3.3 Load Clean Dataset

df = pd.read_csv(r'week2\TAsk_2_2_Pandas\titanic_cleaned.csv')
print("Data Loaded")

# Step 2.3.4 Folder Created


# 1. Cleaned data ko load karna (Path ko folder structure ke mutabiq set kiya hai) [cite: 39]
df = pd.read_csv('week2/TAsk_2_2_Pandas/titanic_cleaned.csv')

target_path = os.path.join('week2/Task_2_3_Visualization', 'visualizations')

# Visualizations folder ki checking aur creation [cite: 40]
if not os.path.exists(target_path):
    os.makedirs(target_path)

# Step 2.3.5 Generate line plot for age distribution

# Plots ka visual style set karna
sns.set_theme(style="whitegrid")

# 2. Line Plot: Age ki distribution dekhne ke liye [cite: 41]
plt.figure(figsize=(10, 6)) # Graph ka size set karna
sns.lineplot(data=df['Age'].value_counts().sort_index()) # Age ke counts ko line mein dikhana
plt.title('Age Distribution Line Plot') # Title dena
plt.savefig(os.path.join(target_path, 'line_plot.png'))
plt.close() # Memory se graph clear karna

# 3. Scatter Plot: Age aur Fare ka muqabla [cite: 42]
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df) # Survived ke hisab se colors dena
plt.title('Age vs Fare Scatter Plot')
plt.savefig(os.path.join(target_path, 'scatter_plot.png'))
plt.close()

# 4. Histogram: Passenger Class (1st, 2nd, 3rd) ki tadad [cite: 43]
plt.figure(figsize=(10, 6))
sns.histplot(df['Pclass'], bins=3, kde=False) # 3 classes ke liye 3 bins
plt.title('Passenger Class Distribution')
plt.savefig(os.path.join(target_path, 'histogram.png')) 
plt.close()

# 5. Bar Chart: Kis class mein survival rate zyada tha [cite: 44]
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Class')
plt.savefig(os.path.join(target_path, 'bar_chart.png')) 
plt.close()

# 6. Box Plot: Fare mein outliers aur distribution dekhne ke liye [cite: 45]
plt.figure(figsize=(10, 6))
sns.boxplot(x='Pclass', y='Fare', data=df)
plt.title('Fare Distribution by Class')
plt.savefig(os.path.join(target_path, 'box_plot.png')) 
plt.close()

# 7. Violin Plot: Gender aur Age ka survival par asar [cite: 46]
plt.figure(figsize=(10, 6))
sns.violinplot(x='Sex', y='Age', hue='Survived', data=df, split=True) # Split se male/female aik hi violin mein aate hain
plt.title('Age Distribution by Gender')
plt.savefig(os.path.join(target_path, 'violin_plot.png')) 

# 8. Heatmap: Columns ke darmiyan correlation (Talluq) [cite: 47]
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=['float64', 'int64']) # Sirf numbers wale columns lena
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm') # Numbers (annotations) show karna
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(target_path, 'heatmap.png')) 
plt.close()

# 9. Pair Plot: Tamam numerical features ka aapsi talluq [cite: 48]
pair_plot = sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived')
pair_plot.savefig(os.path.join(target_path, 'pair_plot.png')) 

print("Task 2.3: Tamam 8 plots kamyabi se 'visualizations' folder mein save ho gaye hain!")