import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

result = pd.read_csv('result.csv')

sns.set(rc={'figure.figsize':(20,20)})
facet = sns.FacetGrid(result, hue="Survived",aspect=5)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, result['Age'].max()))
facet.add_legend()
plt.title('years old')
plt.show() 

