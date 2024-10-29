# ## **Experiment 4**
# ## Using matplotlib and seaborn , draw Line,bar, Histogram, box plot, heatmap , pair plot with respect to Iris dataset.( Attach the python file, and screenshot of all the graph output in a zip file and upload it).



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('IRIS.csv')


df[:]


#Line plot of Sepal Length
plt.figure(figsize=(8,6))
plt.title('Line Plot of Sepal Length')
plt.xlabel('Index')
plt.ylabel('Sepal Length')
plt.plot(df['sepal_length'], color='b')
plt.show()


#Bar Plot of Average Sepal Length by Species
sns.barplot(x='species',y='sepal_length',data=df)
plt.xlabel('Species')
plt.ylabel('Sepal Length')
plt.title('Bar Plot: Sepal Length by Species')
plt.figure(figsize=(8,6))


#Scatterplot (Sepal Length vs Petal Length)
sns.scatterplot(x='sepal_length',y='petal_length',hue='species',data=df)
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Scatterplot (Sepal Length vs Petal Length)')
plt.figure(figsize=(8,6))


#Histogram of Sepal Length
sns.histplot(df['sepal_length'],kde=True,color='g')
plt.title('Histogram of Sepal Length')
plt.xlabel('Sepal Length')
plt.figure(figsize=(8,6))


#Box Plot(Distribution of Sepal Width by Species)
sns.boxplot(x='species',y='sepal_width',data=df)
plt.title('Box Plot (Distribution of Sepal Width by Species)')
plt.xlabel('Species')
plt.ylabel('Sepal Width')
plt.figure(figsize=(8,6))


#Pairplot (Species)
sns.pairplot(df,hue='species',height=3)
sns.set_style('whitegrid')
plt.show()


#Heatmaps
#Heatmap 1
df_numeric_first_10 = df.select_dtypes(include={'float64','int64'}).loc[:9]
plt.title('Heatmap of Iris Dataset')
sns.heatmap(df_numeric_first_10.T,cbar=True,linewidths=(0,5),linecolor='black',vmin=0,vmax=10,xticklabels=[f'Row {i+1}' for i in range(10)],yticklabels=df_numeric_first_10.columns)
plt.figure(figsize=(8,6))


#Heatmap with annot
df_numeric_first_10 = df.select_dtypes(include={'float64','int64'}).loc[:9]
plt.title('Heatmap of Iris Dataset')
sns.heatmap(df_numeric_first_10.T,cbar=True,cmap='winter',linewidths=(0,5),annot=True,linecolor='black',vmin=0,vmax=10,xticklabels=[f'Row {i+1}' for i in range(10)],yticklabels=df_numeric_first_10.columns)
plt.figure(figsize=(8,6))



#Heatmap of Correlation Matrix
df_numeric_first_10 = df.select_dtypes(include={'float64','int64'})
corr=df_numeric_first_10.corr()
plt.title('Correlation Matrix Heatmap')
sns.heatmap(corr.T,cbar=True,cmap='plasma',linewidths=(0,5),annot=True,linecolor='black',vmin=0,vmax=10)
plt.figure(figsize=(8,6))


