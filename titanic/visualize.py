import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def bar_chart(feature, title):
    result = pd.read_csv('result.csv')
    sns.set_style("dark")
    survived = result[result['Survived']==1][feature].value_counts()
    dead = result[result['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    col = df.columns.to_list()
    for count, value in enumerate(col):
            col[count] = feature + '_' + str(value)
    df.columns = col
    df.plot(kind='bar',stacked=True, figsize=(10,10)).legend(loc='upper center', ncol=len(col), title=f"{feature} type")
    plt.title(f'{title}', fontsize=23)
    plt.xticks(rotation=0, horizontalalignment="center")
    plt.xlabel("Survived or Dead", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.savefig(f'/data/{feature}_relation.png')
    # sns.despine(left=True, bottom=True)

def pie_chart(feature, title):
    result = pd.read_csv('result.csv')
    sns.set_style("dark")
    survived = result[result['Survived']==1][feature].value_counts()
    dead = result[result['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    col = df.columns.to_list()
    for count, value in enumerate(col):
            col[count] = feature + '_' + str(value)
    df.columns = col
    df.T.plot(kind='pie', subplots=True, figsize=(20,10), autopct='%.2f', title=f'{title}')
    plt.savefig(f'/data/{feature}_relation.png')
    # sns.despine(left=True, bottom=True)

def plot_corr():
    result = pd.read_csv('result.csv')
    corr = result.corr()
    corr_series = pd.Series(data = corr.iloc[1,2:])
    corr_sort = corr_series.abs().sort_values(ascending = False)
    plt.figure(figsize=(20,10))
    sns.barplot(corr_sort.values, corr_sort.index, orient='h')
    plt.title("Correlation screening based on correlation coefficient")
    plt.savefig('/data/feature_corrlation.png')
    plt.show()

def plot_age():
    result = pd.read_csv('result.csv')
    sns.set(rc={'figure.figsize':(20,20)})
    facet = sns.FacetGrid(result, hue="Survived",aspect=5)
    facet.map(sns.kdeplot,'Age',shade= True)
    facet.set(xlim=(0, result['Age'].max()))
    facet.add_legend()
    plt.title('years old')
    plt.savefig('/data/age_relation.png')
    plt.show() 

def plot_sex_age():
    result = pd.read_csv('result.csv')
    sns.displot(
        result, x="Age", col="Sex", row="Survived",
        binwidth=3, height=3, facet_kws=dict(margin_titles=True),
    )
    plt.savefig('/data/sex_age.png')
    plt.show() 

def main():
    plot_corr()
    bar_chart('Pclass', 'PClass relation with survival')
    pie_chart('Bridge', 'Bridge percentage of survival')
    plot_age()
    plot_sex_age()
