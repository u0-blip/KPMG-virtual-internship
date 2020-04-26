import pandas as pd
import numpy as np


def plot_bars(cats, bar, theme):
    """ the bar need to be nx2 matrix """
    color1 = [1, 0, 0, 0.5]
    color2 = [0, 0, 0, 0.5]

    plt.figure(figsize=[15,10])
    fig, ax1 = plt.subplots()
    num_cat =len(cats.categories)

    plt.bar(range(num_cat), bar[0], color=[color1])
    for i, v in enumerate(bar[0]):
        plt.text(range(num_cat)[i] - 0.25, v + 0.01, str(v))
    title = theme + ' distribution'
    plt.title(title)

    ax1.set_xlabel(theme + ' group')

    plt.xticks(range(len(cats.categories)), cats.categories, rotation=70)

    ax1.set_ylabel('Num of purchase', color=color1)

    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    plt.bar(range(num_cat), bar[1], color=[color2])
    for i, v in enumerate(bar[1]):
        plt.text(range(num_cat)[i] - 0.25, v + 0.01, f"{v:.1f}")
    ax2.set_ylabel('Average value per person', color=color2)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.savefig(title)

def plot_pie(num_cat, labels):
    plt.figure()
    explode = np.zeros_like(num_cat, dtype=np.float)
    # only explode the top 20%
    top_args = np.argsort(-num_cat)[:int(np.ceil(len(num_cat)*0.2))]
    print(top_args)
    explode[top_args] = 0.1
    print(explode)
    plt.pie(num_cat, autopct='%1.1f%%', explode=explode, labels=labels)
    title = 'Pie distribution'
    plt.savefig(title)

def analyse_cat(df, cat_data, theme):
    cats = pd.Categorical(cat_data)

    num_cat = np.array([(cat_data == cat).sum() for cat in cats.categories])
    args = np.argsort(num_cat)[::-1]
    num_cat[::-1].sort()
    print(num_cat)
    cats.categories = cats.categories[args]

    
    purchase = [df[cat_data == cat].past_3_years_bike_related_purchases.astype(float).sum() for cat in cats.categories]
    print(purchase)
    per_capital = np.divide(purchase, num_cat)

    plot_bars(cats, [purchase, per_capital], theme)
    plot_pie(num_cat, cats.categories)

def clean_gender(df):
    for i, p in df.iterrows():
        p = p.gender.strip()
        if p[0] == 'F':
            df.gender[i] = 'F'
        elif p[0] == 'M':
            df.gender[i] = 'M'
        else:
            df.gender[i] = 'U'
    return df

def analyse_interval(df_new, cat_data, theme):
    print(df["DOB"][1].ctime().split(" ")[4])
    df_age = df.dropna(subset=['DOB'])
    lenn = len(df_age)
    df_age["Age"] = 0

    for i in range(1, lenn):
        if i not in df_age.index:
            continue
        if isinstance(df_age.DOB[i], datetime.date):
            tl = len(df_age.DOB[i].ctime().split(" "))
            df_age["Age"][i] = int(2019 - int(df_age.DOB[i].ctime().split(" ")[tl-1]))
        elif isinstance(df_age.DOB[i], str):
            tl = len(df_age.DOB[i].split("-"))
            df_age["Age"][i] = int(2019 - int(df_age.DOB[i].split("-")[tl-1])) 
        if df_age.Age[i] > 100:
            df_age.drop([i], axis=0, inplace=True)


    num_cat = 22
    bins = pd.cut(df_age.Age, num_cat, retbins=True)
    df_age.insert(1, 'age_bin', bins[0])

    num_pur_age = [(df_age.age_bin == cat).sum() for cat in bins[0].values.categories]
    pur_age = [df_age[df_age.age_bin == cat].past_3_years_bike_related_purchases.sum() for cat in bins[0].values.categories]
    pur_age_cap = np.divide(pur_age, num_pur_age)

    color1 = [1, 0, 0, 0.5]
    color2 = [0, 0, 0, 0.5]

    plt.figure()
    fig, ax1 = plt.subplots()

    ax = df_age.Age.plot.hist(bins=22, by="Age", color=[color2])
    plt.title('customer age distribution')

    ax1.set_xlabel('age group')
    ax1.set_ylabel('Num of purchase', color=color1)

    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    plt.bar(np.linspace(0, bins[1][-1], num_cat), pur_age_cap, color=[color1])
    ax2.set_ylabel('Average value per person', color=color2)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color2)

def order_cluster(cluster_field_name, target_field_name,df,ascending):
    # new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final
    
def clean_currency(x):
    if isinstance(x, str):
        return(x.replace('$', '').replace(',', ''))
    return(x)


if __name__ == '__main__':
    analyse_cat(df, df.wealth_segment, 'wealth_segment')
