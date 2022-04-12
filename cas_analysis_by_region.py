"""
Ministry of Transport interview exercise, Senior Data Analyst role
Lucie Jilkova, 12 April 2022

This Script runs some basic analysis and visualisations of the CAS data by region. 
Produces four png plots and an html file with a heatmap.

The script assumes the following files are in the dir_data (./data/) directory:
    ./data/Crash_Analysis_System_(CAS)_data.csv -- CAS csv file as dowloaded from
        https://opendata-nzta.opendata.arcgis.com/datasets/crash-analysis-system-cas-data-1
    ./data/cas_wellington_2019.geojson -- CAS geojson file for Wellington 
        region in 2019
    ./data/Census_Usually_resident_population_count_and_change_by_region_2006_2013_and_2018.csv --
        -- csv data with 2018 Census population count by region, downloaded from
        https://figure.nz/table/HHa5hUtxrCJArkzA

Tested with Python 3.9.7, Pandas 1.3.4, Seaborn 0.11.2, Matplotlib 3.4.3, 
Numpy 1.20.3, Geojson 2.5.0, Folium 0.12.1
"""

import func as fn # functions library in a separate file

def main():
    ### reading CAS data
    print('* Reading CAS data')
    df_cas = fn.get_cas_with_used_columns()
    
    l_severity = ['Non-Injury', 'Minor', 'Serious', 'Fatal']
    
    ### crashes by region in time
    file_crashes_by_region_in_time = 'plt_crashes_by_region_in_time.png'
    print(f'* Plotting crashes by region in time ({file_crashes_by_region_in_time})')
    
    # fatal and injury crashes count 
    f_injury = df_cas['severity'].isin(['Fatal', 'Serious', 'Minor'])
    df_crashes_by_region = df_cas[f_injury][
        ['year', 'region']].value_counts().reset_index(name='count')
    df_crashes_by_region.sort_values(by=['region', 'year'], inplace=True)
    # plot
    fn.plt_lineplot_by_region(df_crashes_by_region, figsize=(10,5), 
                              file_out=file_crashes_by_region_in_time)
    
    ### severity of crashes per capita by region in 2018
    file_severity_per_capita_by_region = 'plt_severity_per_capita_by_region.png'
    print(f'* Plotting severity of crashes per capita by region in 2018 \
    ({file_severity_per_capita_by_region})')
    
    f_2018 = df_cas['year'] == 2018
    df_severity_2018 = df_cas[f_2018][['region', 'severity']].value_counts()
    df_severity_2018 = df_severity_2018.reset_index(name='count')
    df_pop = fn.get_population_by_region()
    df_pop_2018 = df_pop[df_pop['year'] == 2018][['region', 'population']]
    df_severity_2018 = df_severity_2018.merge(df_pop_2018, how='left', on='region')
    df_severity_2018['crashes per capita'] = \
        df_severity_2018['count']/df_severity_2018['population']*100000
    df_sev_pop_2018 = df_severity_2018.pivot('region', 'severity', 'crashes per capita')
    df_sev_pop_2018 = df_sev_pop_2018[l_severity]
    # plot
    fn.plt_severity_per_capita(df_sev_pop_2018, figsize=(6,6), 
                               file_out=file_severity_per_capita_by_region)
    
    ### severity distributions and open vs. urban road crashes by region
    file_severity_and_open_by_region = 'plt_severity_and_open_by_region.png'
    print(f'* Plotting severity distributions and open vs. urban road crashes by region \
    ({file_severity_and_open_by_region})')
        
    # use data from years 2016--2020
    f_years = df_cas['year'].between(2016, 2020)
    # severity by region
    df_severity_rel, df_severity_dist = fn.relative_pivot(df_cas, 'severity', filter_data=f_years)
    df_severity_rel = df_severity_rel[l_severity]
    # urban vs. open by region
    df_uo_rel, df_uo_dis = fn.relative_pivot(df_cas, 'urban', filter_data=f_years)
    # join to a single DF
    df_severity_uo_rel = df_severity_rel.join(df_uo_rel)
    # sort count distribution according to relative distribution
    df_severity_uo_rel.sort_values(by='Non-Injury', inplace=True)
    df_severity_dist = df_severity_dist.reindex(index=df_severity_uo_rel.index)
    # plot
    fn.plt_bar_dist_by_region(df_severity_uo_rel, 
                              df_values=df_severity_dist, 
                              figsize=(10, 6), 
                              file_out=file_severity_and_open_by_region)
    
    ### severity distributions for open vs. urban crashes
    f_severity_of_uo = 'plt_severity_of_uo.png'
    print(f'* Plotting severity distributions for open vs. urban crashes \
    ({f_severity_of_uo})')
    
    # use data from yers 2016--2020
    df_uo_severity = df_cas[f_years][['urban', 'severity']].value_counts()
    df_uo_severity = df_uo_severity.reset_index(name='count')
    # total numeber of open and urban crashes
    total_open = df_uo_severity.groupby('urban').sum().loc['Open'][0]
    total_urban = df_uo_severity.groupby('urban').sum().loc['Urban'][0]
    # calculate relative percentage of severity for open and urban
    f_open = df_uo_severity['urban'] == 'Open'
    df_uo_severity['percent_uo'] = 0.0 
    df_uo_severity.loc[f_open, 'percent_uo'] = \
        df_uo_severity.loc[f_open, 'count']/total_open*100
    df_uo_severity.loc[~f_open, 'percent_uo'] = \
        df_uo_severity.loc[~f_open, 'count']/total_urban*100
    # plot
    fn.plt_dist_uo_severity(df_uo_severity, 
                            figsize=(7, 5), 
                            file_out=f_severity_of_uo)
    
    ### generate heatmap of 2018 Wellington region crashes
    file_heatmap = 'heatmap_2019_wellington.html'
    print(f'* Generating heatmap of 2018 Wellington region crashes ({file_heatmap})')
    df_well_2018 = fn.get_coordinates_from_geojson()
    fn.plt_heatmap(df_well_2018, file_out=file_heatmap)

if __name__ == '__main__':
    main()