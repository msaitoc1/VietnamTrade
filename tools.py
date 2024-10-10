#Datascience methods used in trade research
import pandas as pd
import os
import doctest
import seaborn as sns
import matplotlib.pyplot as plt

def census_df(file_path, port):
    """
    Creates a df using a csv downloaded from usatrade.census.gov
    Cleans the values traded by removing commas and converting to float
    Removes the "through MONTH" at the end of the most recent year's data.
    
    Parameters:
    file_path: a csv downloaded from usatrade.census.gov
    port: boolean representing if the data is port data
    
    Returns: 
    pd.DataFrame: Cleaned Dataframe


    creates a df where all the column names are years in int form
    
    """
    with open(file_path, 'r') as file:
        for idx, line in enumerate(file):
            if "Commodity" in line:
                header_row = idx
                break 
    if port == True:
        val_col = "Customs Value (Gen) ($US)"
    else: 
        val_col = "Total Value ($US)"
    df = pd.read_csv(file_path, skiprows=header_row)
    df = df.rename(columns={val_col:'Value'})
    df['Time'] = df['Time'].str.replace(r'(\d{4}) through [A-Za-z]+', r'\1', regex=True) 
    #df['Time'] = df['Time'].astype(int)
    df.fillna(0)
    df["Value"] = df["Value"].astype(str).str.replace(',', '').astype(float)
    
    return df 

def census_pivot(df, codes_only= None):
    """
    Creates a new df with years as columns and amount of commodity traded as values.
    Converts value exported/imported to millions of dollars.
    Optionally removes the descriptions from the end of the HS code.
    
    Parameters:
    df (pd.DataFrame): a df containing data from usatrade.census.gov assumes there is only ONE
    column for each commodity (i.e there is only one destination or origin)
    codes_only (int): Represents the digits of the HS codes. 
        The commodity column is updated to leave only HS codes.

    Returns: 
    pd.DataFrame: Aggregated dataframe

    >>> import pandas as pd
    >>> data = {'Commodity':['64 Bovine meat', '65 Bovine Meat', '66 Bovine Meat'],
    ...        'Time': [2023,2023,2024], 'Customs Value (Gen) ($US)':[1000, 2500, 3000]}
    >>> df = pd.DataFrame(data)
    >>> result = census_pivot(df, True, 2)
    >>> result  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Time          2023    2024
    Commodity
    64         0.00010  0.0000
    65         0.00025  0.0000
    66         0.00000  0.0003
    """
    if codes_only:
        df["Commodity"] = df["Commodity"].str[0:codes_only]
    
    df = df.loc[:, ["Commodity", "Time", "Value"]]
    df["Value"] = df["Value"]/10000000
    pivoted_df = df.pivot(index="Commodity", columns = "Time", values = "Value")
    return pivoted_df.fillna(0)


def addTotalCol(df):
    df["Total"] = df.sum(axis=1)
    return df

def mergeLocations(df):
    """
    Takes an unpivoted dataframe and merges multiple locations by summing the values for each product and year.
    If port == True, changes column name for port specific data.

    Parameters:
    df (pd.DataFrame): unpivoted dataframe from usatrade.census.gov

    Returns:
    df (pd.DataFrame): merged Dataframe

    >>> import pandas as pd
    >>> data = {'Commodity':['65 Bovine Meat', '65 Bovine Meat', '66 Bovine Meat'],
    ...        'Time': [2023,2023,2024], 'Location':['Seattle', 'Seattle-Tacoma', 'Seattle'], 'Customs Value (Gen) ($US)':[1000, 2500, 3000]}
    >>> df = pd.DataFrame(data)
    >>> result = mergeLocations(df, True)
    >>> result # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
           Commodity  Time  Customs Value (Gen) ($US)
    0 65 Bovine Meat  2023                      3500
    1 66 Bovine Meat  2024                      3000

    """
    df.loc[:, "Value"] = df["Value"].fillna(0).astype(float)
    aggregated_df = df.groupby(['Commodity', 'Time'], as_index=False)["Value"].sum()
    return aggregated_df

def percentsDf_totalTrade(df_specific, df_total):
    """
    Given two pivoted dataframes, one representing a specific subsection of trade and the other representing total trade,
    returns a new df of the percent of total trade that each subsection value represents.

    Parameters:
    df_specific (pd.DataFrame): subsection of total trade data
    df_total (pd.DataFrame): total aggregated trade data
    
    Returns:
    df (pd.DataFrame): percents of total trade dataframe

    >>> import pandas as pd
    >>> data_part = {'Commodity':[63,64,65], '2022':[100, 200, 300], '2023':[200, 300, 400], '2024':[200,300,400]}
    >>> data_whole =  {'Commodity':[63,64,65,66], '2022':[400, 100, 600, 500], '2023':[300, 300, 600, 100], '2024':[200,400,400, 0]}
    >>> df_part = pd.DataFrame(data_part)
    >>> df_total = pd.DataFrame(data_whole)
    >>> df_part = df_part.set_index("Commodity")
    >>> df_total = df_total.set_index("Commodity")
    >>> result = percentsDf_totalTrade(df_part, df_total)
    >>> result # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
               2022      2023  2024
    Commodity
    63         0.25  0.666667  1.00
    64         2.00  1.000000  0.75
    65         0.50  0.666667  1.00

    """
    if df_total.shape[0] > df_specific.shape[0]:
        remove_rows = set(df_total.index)-set(df_specific.index)
        matched_total_df = df_total.drop(remove_rows, axis = 0)
    elif df_total.shape[0] < df_specific.shape[0]:
        raise ValueError("df_total has fewer rows than df_specific. Cannot proceed with the operation.") 
    if matched_total_df.shape[1] > df_specific.shape[1]:
        remove_cols = set(matched_total_df.columns) - set(df_specific.columns)
        matched_total_df = matched_total_df.drop(remove_cols, axis =1)
    elif matched_total_df.shape[1] < df_specific.shape[1]:
        raise ValueError("df_total has fewer columns than df_specific. Cannot proceed with the operation.") 
    pcts_df = df_specific.div(matched_total_df)
    return pcts_df

def percentsDf_yearlyCat(df):
    """
    Given one dataframe, finds the percentage that each category represents of total trade, and returns a df
    of these values. Percents are given as x.xx% form
    """
    df.loc["sums"] = df.sum()
    cleaned_agg_df_pcts = df.div(df.loc['sums'], axis=1) * 100
    return cleaned_agg_df_pcts.drop("sums")

def save_csv(df, name, folder_path):
    """
    saves file to desired folder with a given name. 
    If the folder doesn't exist, one is created. 
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filepath = os.path.join(folder_path, name + ".csv")
    df.to_csv(filepath)
    print(f"DataFrame saved to {filepath}")

def plot_pcts_bargraph(series, title, xlab, ylab, fileName=None, folderPath = None):
    """
    Plots percentages 
    """
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot( x=series.index, y=series.values)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xticks(rotation =45)
    if fileName and folderPath:
        path = os.path.join(folderPath, fileName +".png")
        plt.savefig(path)
    plt.show()  

def plot_pcts_linegraph(series, title, xlab, ylab, fileName=None, folderPath=None):
    """
    Plots percentages 
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot( x=series.index, y=series.values, palette='Blues')
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xticks(rotation =45)
    if fileName and folderPath:
        path = os.path.join(folderPath, fileName +".png")
        plt.savefig(path)
    plt.show()  

def plot_stacked_bargraph(df, title, xlab, ylab, fileName=None, folderPath=None):
    """
    Using percents df which has years as cols and parts of an total as rows, creates
    a stacked bar graph with a given title and x and y labels

    Parameters:
    df (pandas.DataFrame): uscensus.gov df pivoted and with values as percents
    title (str): title of the bar graph
    xlab (str): label on x axis
    ylab (str): label on y axis
    fileName (str, default = None): name of the file if plot should be saved
    folderPath (str, default = None): name of the folder path to save the file to
    
    Returns:
    view of plot
    """
    ax = df.plot(kind='bar', stacked=True)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    if folderPath and fileName: 
        plt.savefig(os.path.join(folderPath, fileName+".png"))
    plt.show()
def plot_multiple_lines(df, title, xlab, ylab, fileName=None, folderPath=None):
    """
    Given a dataframe of values with cols as year, plots each row as a line on a line graph. 
    Optionally saves to desired location. Both fileName and folderPath must be given to save.

    Parameters:
    df (Pandas.DataFrame): dataframe with cols as years
    title (str): Title of the line graph
    xlab (str): label for x axis
    ylab (str): label for y axis
    fileName (str) default = None: optional name for saving the png to computer
    folderPath (str) default = None: optional folder path for where to save png

    Returns:
    view of plt
    """
    ax = df.plot(kind='line')
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    if fileName and folderPath:
        plt.savefig(os.path.join(folderPath, fileName+".png"))
    plt.show()


if __name__ == "__main__":
    import doctest
    doctest.testmod()

