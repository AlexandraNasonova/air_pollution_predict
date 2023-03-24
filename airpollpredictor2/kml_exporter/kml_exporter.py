import datetime
import pandas as pd

### KML exporter exports information about stations in kml format
### Requires kml_elements.csv and final_metadata.csv files
### Can be used to export stations corresponding to specific country or city
### If you filter out unnecesary stations from final_metadata.csv
### Each country has subfolders corresponding to 3 categories: inactive, active, active with unspecified location
### Can be modified and tweaked to group by something else, if no extra subfolder levels are required
### If you need extra subfolder levels you need to edit and add new kml templates into kml_elements.csv

def kml_exporter(df, Username='Username', Githublink='https://github.com/Username') -> None:
    kml_df = pd.read_csv('kml_elements.csv')
    folder_list = []
    stat_count = 0
    df1 = df.drop_duplicates(['AirQualityStation'], keep='first')
    for element in list(df1['Countrycode'].unique()):
        dff = df1[df1['Countrycode'] ==element].reset_index(drop=True)
        temp_page_inactive_list = []
        temp_page_active_list = []
        temp_page_new_list = []
        for i in range(len(list(dff['AirQualityStation']))):
            DateEnd = ''
            a_status = ''
            AirQualityStation = dff.loc[i, 'AirQualityStation']
            LocalCode = dff.loc[i, 'LocalCode']
            StationName = dff.loc[i, 'StationName']
            try:
                ObservationDateBegin = dff.loc[i, 'ObservationDateBegin'][0:10]
            except:
                ObservationDateBegin = ''
            status1 = [dff.loc[i, 'ObservationDateEnd']!=dff.loc[i, 'ObservationDateEnd']][0]
            status2 = [dff.loc[i, 'LocalCode']!=dff.loc[i, 'LocalCode']][0]
            if (status1==True) or (status2==True):
                try:
                    DateEnd = dff.loc[i, 'ObservationDateEnd'][0:10]
                except:
                    DateEnd = ''
                a_status = 'inactive station'
            elif dff.loc[i, 'StationCity']!=dff.loc[i, 'StationCity']:
                a_status = 'active station without location'
                DateEnd = ''
            else:
                a_status = 'active station'
                DateEnd = ''
            ObservationDateEnd = DateEnd
            active_status = a_status
            AirQualityStationType = dff.loc[i, 'AirQualityStationType']
            AirQualityStationArea = dff.loc[i, 'AirQualityStationArea']
            Longitude = format(dff.loc[i, 'Longitude'], '.6f')
            Latitude = format(dff.loc[i, 'Latitude'], '.6f')
            Altitude = format(dff.loc[i, 'Altitude'])
            my_str = (kml_df['val'][0]).format(AirQualityStation=AirQualityStation, \
                                               LocalCode=LocalCode, StationName=StationName, \
                                               ObservationDateBegin=ObservationDateBegin, \
                                               ObservationDateEnd=ObservationDateEnd, \
                                               active_status=active_status, \
                                               AirQualityStationType=AirQualityStationType, \
                                               AirQualityStationArea=AirQualityStationArea, \
                                               Longitude=Longitude, Latitude=Latitude, Altitude=Altitude)
            if active_status== 'inactive station':
                temp_page_inactive_list.append(my_str)
            elif active_status== 'active station':
                temp_page_active_list.append(my_str)
            elif active_status== 'active station without location':
                temp_page_new_list.append(my_str)
        subfolder_text1 = kml_df['val'][3] + ''.join(temp_page_inactive_list) + kml_df['val'][4]
        subfolder_text1 = subfolder_text1.format(aa_status='inactive', StatusNumber=len(temp_page_inactive_list))
        subfolder_text2 = kml_df['val'][3] + ''.join(temp_page_active_list) + kml_df['val'][4]
        subfolder_text2 = subfolder_text2.format(aa_status='active', StatusNumber=len(temp_page_active_list))
        subfolder_text3 = kml_df['val'][3] + ''.join(temp_page_new_list) + kml_df['val'][4]
        subfolder_text3 = subfolder_text3.format(aa_status='active_unspecified_location', StatusNumber=len(temp_page_new_list))
        subfolder_text_final = subfolder_text1 + subfolder_text2 + subfolder_text3
        StationsNumber = len(temp_page_inactive_list)+len(temp_page_active_list)+len(temp_page_new_list)
        stat_count = stat_count + StationsNumber
        folder_text = kml_df['val'][1] + subfolder_text_final + kml_df['val'][2]
        folder_text = folder_text.format(Countrycode=element, StationsNumber=StationsNumber)
        folder_list.append(folder_text)
    document_text = kml_df['val'][5] + ''.join(folder_list) + kml_df['val'][6]
    TotalStationsNum = stat_count
    Date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    UserName, GithubLink = Username, Githublink
    document_text = document_text.format(TotalStationsNum=TotalStationsNum, Date=Date, \
                                         UserName=UserName, GithubLink=GithubLink)
    filename = datetime.datetime.now().strftime('%Y_%m_%d_%H') + '_' + 'stations' + '.kml'
    with open(filename, 'w', encoding='utf-8') as the_file:
        the_file.write(document_text)

if __name__ == '__main__':
    kml_exporter(pd.read_csv('final_metadata.csv', low_memory=False))