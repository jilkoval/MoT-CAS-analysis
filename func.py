"""
Various helper functions (read data, plotting, repeated data manipulation...)
"""

import pandas as pd
import seaborn as sns
import numpy as np
import geojson
import folium
from matplotlib import pyplot as plt
from folium.plugins import HeatMap

# set pytplot style for plotting, adopting seaborn-talk with dark grid style
plt.style.use('default')
plt.style.use('seaborn-darkgrid')
plt.style.use('seaborn-talk')

# directory where data files are stored
dir_data = './data/'

def get_cas_with_used_columns(file='Crash_Analysis_System_(CAS)_data.csv',
                              columns=[
                                    # 'X',
                                    # 'Y', 
                                    'OBJECTID', 
                                    # 'advisorySpeed', 
                                    # 'areaUnitID', 
                                    # 'bicycle', 
                                    # 'bridge', 
                                    # 'bus', 
                                    # 'carStationWagon', 
                                    # 'cliffBank', 
                                    # 'crashDirectionDescription', 
                                    # 'crashFinancialYear', 
                                    # 'crashLocation1', 
                                    # 'crashLocation2', 
                                    # 'crashRoadSideRoad', 
                                    'crashSeverity', 
                                    # 'crashSHDescription', 
                                    'crashYear', 
                                    # 'debris', 
                                    # 'directionRoleDescription', 
                                    # 'ditch', 
                                    # 'fatalCount', 
                                    # 'fence', 
                                    # 'flatHill', 
                                    # 'guardRail', 
                                    # 'holiday', 
                                    # 'houseOrBuilding', 
                                    # 'intersection', 
                                    # 'kerb', 
                                    # 'light', coordinate
                                    # 'meshblockId', 
                                    # 'minorInjuryCount', 
                                    # 'moped', 
                                    # 'motorcycle', 
                                    # 'NumberOfLanes', 
                                    # 'objectThrownOrDropped', 
                                    # 'otherObject', 
                                    # 'otherVehicleType', 
                                    # 'overBank', 
                                    # 'parkedVehicle', 
                                    # 'pedestrian', 
                                    # 'phoneBoxEtc', 
                                    # 'postOrPole', 
                                    'region', 
                                    # 'roadCharacter', 
                                    # 'roadLane', 
                                    # 'roadSurface', 
                                    # 'roadworks', 
                                    # 'schoolBus', 
                                    # 'seriousInjuryCount', 
                                    # 'slipOrFlood', 
                                    # 'speedLimit', 
                                    # 'strayAnimal', 
                                    # 'streetLight', 
                                    # 'suv', 
                                    # 'taxi', 
                                    # 'temporarySpeedLimit', 
                                    # 'tlaId', 
                                    # 'tlaName', 
                                    # 'trafficControl', 
                                    # 'trafficIsland', 
                                    # 'trafficSign', 
                                    # 'train', 
                                    # 'tree', 
                                    # 'truck', 
                                    # 'unknownVehicleType', 
                                    'urban', 
                                    # 'vanOrUtility', 
                                    # 'vehicle', 
                                    # 'waterRiver', 
                                    # 'weatherA', 
                                    # 'weatherB'
                                    ]):
    """
    Reads columns used in the analysis from a sepcified csv file.
    - Assumes columns OBJECTID, crashSeverity, CrashYear, region, and urban are
    present in the file.
    - Renames some columns for convenience.
    - Sets index to OBJECTID.
    - Removes strings ' Region' and ' Crash' from region and crashSeverity columns.
    - Returns pandas DF with columns [severiy, year, region, urban].
    """
    df = pd.read_csv(dir_data+file, usecols=columns)
    
    df.rename(columns={'crashYear': 'year', 'crashSeverity': 'severity'}, inplace=True)
    df.set_index('OBJECTID', inplace=True)
    
    # filter out crashes with no region (for a complete anlysis, missing values
    #    should be figured out from other columns)
    f_no_region = df.region.isnull()
    print(f'! There are {f_no_region.sum():,} (out of {df.shape[0]:,}) crashes with a missing region value.\
    \n  ! Not using those in the further analysis.\
    \n  ! For a complete anlysis, missing values should be figured out from other columns if possible.')
    df = df[~f_no_region].copy()
    df['region'] = df['region'].apply(lambda x: x.replace('Region', '').strip())
    df['severity'] = df['severity'].apply(lambda x: x.replace('Crash', '').strip())
    
    return df

def get_population_by_region(
        file='Census_Usually_resident_population_count_and_change_by_region_2006_2013_and_2018.csv'):
    """
    Reads population by region and year from given file (census years 2006, 2013, and 2018).
    Source: https://figure.nz/table/HHa5hUtxrCJArkzA
    Returns DF with columns [year, region, population].
    """
    df = pd.read_csv(dir_data+file, usecols=['Census year', 'Region', 'Value'])
    df = df[df['Census year'].isin(['2006','2013','2018'])]
    df = df.astype({'Census year': 'int32', 'Region': 'str', 'Value': 'int32'})
    df.rename(columns={'Census year': 'year',
                       'Region': 'region',
                       'Value': 'population'}, inplace=True)
    return df

def plt_lineplot_by_region(df, figsize=(10,5),
                           file_out=None):
    """
    Plots crashes count in time lineplot.
    Input
        - df: pandas DF with the counts of crashes by region. Columns 
            'region', 'year', 'crash count' are assumed.
        - figsize: tupple, optional, output figure size in inches.
        - file_out: str or None, optional. Output file. 
            If None, the result is plotted on screen. Otherwise it is
            saved as file_out. The default is None.
    Returns matplotlib Figure.
    """
    fig = plt.figure(figsize=figsize)
    
    # paramters of the lineplot
    arg_all = {'alpha':0.8, 'lw':2.5}
    
    # plot South Island by dashed lines
    dict_regions = get_regions_island()
    dict_ls = {'n': '-', 's': '--'}    
    
    for region, island in dict_regions.items():
        f_r = df['region'] == region
        plt.plot(df[f_r]['year'], df[f_r]['count'],
                 label=region,
                 ls=dict_ls[island],
                 **arg_all)
    
    # labels and legend
    plt.xlabel('year')
    plt.ylabel('fatal and injury crashes count')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    
    plt.tight_layout(pad=0.2)
    plot_output(file_out=file_out)
    return fig

def plt_severity_per_capita(df, figsize=(6,6), file_out=None, title=None):
    """
    Plots a colored table of crashes per capita by severity and by region. 
    Input
        - df: pandas DF crashes per capita. 
            Index 'region' is assumed.
            Columns 'Non-Injury', 'Minor', 'Serious', 'Fatal' are assumed.
        - figsize: tupple, optional, output figure size in inches.
        - file_out: str or None, optional. Output file. 
            If None, the result is plotted on screen. Otherwise it is
            saved as file_out. The default is None.
        - title: title of the plot. No title if None (default).
    Returns matplotlib Figure.
    """
    # the table is formed by four individual heatmap plots for each column
    fig, axes = plt.subplots(
        nrows=1, ncols=4, figsize=figsize, sharey=True)
    
    # colorscale
    cmap = 'YlOrRd'
    
    # sort by non-injury values
    df_ = df.sort_values(by='Non-Injury')
    
    # loop through DF columns, plot each column as a separate heatmap 
    for i, sev in enumerate(df.columns):
        df_i = pd.DataFrame(df_[sev])
        sns.heatmap(df_i, annot=True, fmt='.0f', ax=axes[i], cmap=cmap, cbar=False)
        if i>0:
            axes[i].set_ylabel(None)
            
    if title is not None:
        axes[1].set_title('Crashes per 100,000 population by region in 2018', pad=20)
    
    plt.tight_layout(pad=0.2)
    plot_output(file_out=file_out)
    return fig
    
def plt_bar_dist_by_region(df, df_values=None, figsize=(10, 6), file_out=None):
    """
    Plots a severity and (open vs. urban) horizontal bar charts by region 
    with a common region axis.
    Input
        - df: DF with the percentage of crashes by severity and (open vs. urban) for 
            regions. Assumed structure: index=region, 
            columns=['Minor', 'Serious', 'Fatal', 'Open'].
        - df_values: DF or None, optional. 
            Contains the absolute numbers of crashes by severity for regions.
            Assumed structure: index=region, columns=['Minor', 'Serious', 'Fatal'].
            If df_values is not None, numbers are used for annotaion.
            If df_values is None, no annotaion is used. Default is None.
        - figsize: tupple, optional, output figure size in inches.
        - file_out: str or None, optional
            Output file. If None, the result is plotted on screen. Otherwise it is
            saved as file_out. The default is None.
    Returns matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True,
                                   gridspec_kw={'width_ratios': [2, 1]})
    
    # colors and parameters
    severity_colors = plt.get_cmap('YlOrRd')(np.linspace(0.4, 0.95, 3))
    bar_width = 0.7
    
    # plot
    l_columns = ['Minor', 'Serious', 'Fatal']
    df[l_columns].plot.barh(
        stacked=True, ax=ax1, color=severity_colors, width=bar_width)
    df['Open'].plot.barh(ax=ax2, width=bar_width)
    
    # anotation
    if df_values is not None:
        values = df_values[l_columns].values.flatten("F")
        for i, p in enumerate(ax1.patches):
            left, bottom, width, height = p.get_bbox().bounds
            if i<32:
                xpos = left + width/2
                dict_par = {'ha': 'center'}
            else:
                xpos = left + width + 0.3
                dict_par = {'ha': 'left', 'color': severity_colors[2]}
            ax1.annotate(str(values[i]), xy=(xpos, bottom+height/2), 
                         va='center', **dict_par)
    
    # labels and legend
    ax1.set_title('% of total crashes', pad=25)
    ax1.set_xlabel('% of total crashes')
    ax1.legend(bbox_to_anchor=(0, 0.97), loc='lower left', ncol=3)
    
    ax2.set_title('% of open road crashes')
    ax2.set_xlabel('% of open road crashes')

    plt.tight_layout(pad=0.2)
    plot_output(file_out=file_out)
    return fig

def plt_dist_uo_severity(df, figsize=(7, 5), title=None, file_out=None):
    """
    Plots a barchart of severity distribution of urban and open crashes.
    Input
        - df: pandas DF. 
            Columns 'urban', 'severity', 'count' and 'percent_uo' are assumed.
        - figsize: tupple, optional, output figure size in inches.
        - file_out: str or None, optional. Output file. 
            If None, the result is plotted on screen. Otherwise it is
            saved as file_out. The default is None.
        - title: title of the plot. No title if None (default).
    Returns matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # x axis ticks and bar width
    labels = df['severity'].unique().tolist()
    x = np.arange(len(labels))
    width = 0.4
  
    f_open = df['urban'] == 'Open'
    
    # include total number of crashes in the label
    n_open = df[f_open]['count'].sum()
    n_urban = df[~f_open]['count'].sum()
    
    ax.bar(x - width/2, df[f_open]['percent_uo'], width, 
           label=f'open ({n_open:,} total)')
    ax.bar(x + width/2, df[~f_open]['percent_uo'], width, 
           label=f'urban ({n_urban:,} total)')
    
    # labels
    ax.set_ylabel('%')
    if title is not None:
        ax.set_title('Severity of open vs. urban crashes')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout(pad=0.2)
    plot_output(file_out=file_out)
    return fig
    
def plot_output(file_out=None):
    """
    Sets pyplot output
    file_out: str or None, optional
        Output file for pyplot. If None, the result is plotted on screen. 
        Otherwise it is saved as fout. 
        The default is None.
    """
    if file_out is not None:
        plt.savefig(file_out)
    else:
        plt.show()

def get_regions_island():
    """
    Helper function to sort regions from North to South and assign
    North / South Island labels (n or s).
    Returns a dict where keys are regions and items are 'n' or 's'.
    """
    l_ni = ['Northland', 'Auckland', 'Waikato', 'Bay of Plenty',
            'Gisborne', "Hawke's Bay", 'Taranaki', 'Manawatū-Whanganui', 
            'Wellington']
    l_si = ['Tasman', 'Nelson', 'Marlborough' , 'West Coast', 'Canterbury', 
            'Otago', 'Southland']
    return {r: 'n' if r in l_ni else 's' for r in l_ni+l_si}

def get_coordinates_from_geojson(file='./data/cas_wellington_2019.geojson'):
    """
    Reads a geojson file and extracts latitude and longitude values.
    Returns DF with OBJECTID as index and lat, lon as columns.
    """
    with open('./data/cas_wellington_2019.geojson') as f:
        gj = geojson.load(f)
    df_gj = pd.DataFrame(
        [[x['properties']['OBJECTID'], 
          x['geometry']['coordinates'][1], 
          x['geometry']['coordinates'][0]] for x in gj['features']],
        columns=['OBJECTID', 'lat', 'lon'])
    return df_gj.set_index('OBJECTID')

def plt_heatmap(df, center=[-41.28386280760349, 174.7786527149068],
                file_out='heatmap_2019_wellington.html'):
    """
    Creates folium heatmap of crashes.
    Input
        - df: DF with latitude and longitude coordinates.
    Returns html centered on center coordinates.
    """
    base_map = folium.Map(location=center, zoom_start=13, tiles='cartodbpositron')
    l_heat = df.values.tolist()
    HeatMap(data=l_heat,
            gradient={0.1:'yellow', 0.5:'orange', 1.0:'red'}, 
            radius=10, 
            blur=10).add_to(base_map)
    base_map.save(file_out)
      
def relative_pivot(df, column_to_pivot, filter_data=None, index='region'):
    """
    Filters, counts and pivots dataframe for given column.
    - df: DF with items to be counted.
    - column_to_pivot: str, column to pivot.
    - filter_data: bool series of the same dimension as DF (same number of rows)
        that is applied on the data before counting values. If None, the full 
        DF is used. Default is None
    - index: str, column to be used as index in the pivoted data. 
        Default is 'region'.
    Returns two DFs, both with given index and columns of distinct values in column_to_pivot.
    - The 1st DF with relative (%) distributions (row-wise).
    - The 2nd DF with count values.
    """
    if filter_data is not None:
        df_counts = df[filter_data][[index, column_to_pivot]].value_counts()
    df_counts = df_counts.reset_index(name='count')
    df_dist = df_counts.pivot(index=index, columns=column_to_pivot)
    df_dist.columns = df_dist.columns.droplevel()
    df_rel = df_dist.apply(lambda x: x[:]/x.sum()*100, axis=1)
    return df_rel, df_dist