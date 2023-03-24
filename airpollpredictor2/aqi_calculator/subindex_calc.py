import numpy as np
if __name__ == "__main__":
    import pandas as pd


### AQI subindex calculator for SO2
def aqi_easy_1(df, column_index): # SO2
    conc = [j for i in df.iloc[:, column_index:(column_index+1)].values.tolist() for j in i]
    c_list = []
    aqi = 0
    c = 0
    for i in range(len(conc)):
        try:
            if conc[i]<=35:
                aqi = int(round((50 * conc[i]) / 35, 0))
            elif 35<conc[i]<=75:
                aqi = int(round((100 - 51) * (conc[i] - 36) / (75 - 36) + 51, 0))
            elif 100<conc[i]<=185:
                aqi = int(round((150 - 101) * (conc[i] - 76) / (185 - 76) + 101, 0))
            elif 185<conc[i]<=304:
                aqi = int(round((200 - 151) * (conc[i] - 186) / (304 - 186) + 151, 0))
            elif 304<conc[i]<=604:
                aqi = int(round((300 - 201) * (conc[i] - 305) / (604 - 305) + 201, 0))
            elif 604<conc[i]<=804:
                aqi = int(round((400 - 301) * (conc[i] - 605) / (804 - 605) + 301, 0))
            else:
                aqi = int(round((500 - 401) * (conc[i] - 805) / (1004 - 805) + 401, 0))
        except:
            aqi = np.nan
        c_list.append(aqi)
    return c_list

### AQI subindex calculator for PM10
def aqi_easy_5(df, column_index): # PM10
    conc = [j for i in df.iloc[:, column_index:(column_index+1)].values.tolist() for j in i]
    c_list = []
    aqi = 0
    c = 0
    for i in range(len(conc)):
        try:
            if conc[i]<=54:
                aqi = int(round((50 * conc[i]) / 54, 0))
            elif 54<conc[i]<=154:
                aqi = int(round((100 - 51) * (conc[i] - 55) / (154 - 55) + 51, 0))
            elif 154<conc[i]<=254:
                aqi = int(round((150 - 101) * (conc[i] - 155) / (254 - 155) + 101, 0))
            elif 254<conc[i]<=354:
                aqi = int(round((200 - 151) * (conc[i] - 255) / (354 - 255) + 151, 0))
            elif 354<conc[i]<=424:
                aqi = int(round((300 - 201) * (conc[i] - 355) / (424 - 355) + 201, 0))
            elif 424<conc[i]<=504:
                aqi = int(round((400 - 301) * (conc[i] - 425) / (504 - 425) + 301, 0))
            else:
                aqi = int(round((500 - 401) * (conc[i] - 505) / (604 - 505) + 401, 0))
        except:
            aqi = np.nan
        c_list.append(aqi)
    return c_list

### AQI subindex calculator for O3 8Hour
def aqi_easy_7_8H(df, column_index): # O3 8Hour
    conc = [j for i in df.iloc[:, column_index:(column_index+1)].values.tolist() for j in i]
    c_list = []
    aqi = 0
    c = 0
    for i in range(len(conc)):
        try:
            if conc[i]<=0.054:
                aqi = int(round((50 * conc[i]) / 0.054, 0))
            elif 0.054<conc[i]<=0.070:
                aqi = int(round((100 - 51) * (conc[i] - 0.055) / (0.070 - 0.055) + 51, 0))
            elif 0.070<conc[i]<=0.085:
                aqi = int(round((150 - 101) * (conc[i] - 0.071) / (0.085 - 0.071) + 101, 0))
            elif 0.085<conc[i]<=0.105:
                aqi = int(round((200 - 151) * (conc[i] - 0.086) / (0.105 - 0.086) + 151, 0))
            elif 0.105<conc[i]<=0.200:
                aqi = int(round((300 - 201) * (conc[i] - 0.106) / (0.200 - 0.106) + 201, 0))
            else:
                aqi = np.nan
        except:
            aqi = np.nan
        c_list.append(aqi)
    return c_list

### AQI subindex calculator for O3 1Hour
def aqi_easy_7_1H(df, column_index): # O3 1Hour
    conc = [j for i in df.iloc[:, column_index:(column_index+1)].values.tolist() for j in i]
    c_list = []
    aqi = 0
    c = 0
    for i in range(len(conc)):
        try:
            if 0.124<conc[i]<=0.164:
                aqi = int(round((150 - 101) * (conc[i] - 0.125) / (0.164 - 0.125) + 101, 0))
            elif 0.164<conc[i]<=0.204:
                aqi = int(round((200 - 151) * (conc[i] - 0.165) / (0.204 - 0.165) + 151, 0))
            elif 0.204<conc[i]<=0.404:
                aqi = int(round((300 - 201) * (conc[i] - 0.205) / (0.404 - 0.205) + 201, 0))
            elif 0.404<conc[i]<=0.504:
                aqi = int(round((400 - 301) * (conc[i] - 0.405) / (0.504 - 0.405) + 301, 0))
            elif 0.504<conc[i]<=0.604:
                aqi = int(round((500 - 401) * (conc[i] - 0.505) / (0.604 - 0.505) + 401, 0))
            else:
                aqi = np.nan
        except:
            aqi = np.nan
        c_list.append(aqi)
    return c_list

### AQI subindex calculator for NO2
def aqi_easy_8(df, column_index): # NO2
    conc = [j for i in df.iloc[:, column_index:(column_index+1)].values.tolist() for j in i]
    c_list = []
    aqi = 0
    c = 0
    for i in range(len(conc)):
        try:
            if conc[i]<=53.0:
                aqi = int(round((50 * conc[i]) / 53, 0))
            elif 53<conc[i]<=100:
                aqi = int(round((100 - 51) * (conc[i] - 54) / (100 - 54) + 51, 0))
            elif 100<conc[i]<=360:
                aqi = int(round((150 - 101) * (conc[i] - 101) / (360 - 101) + 101, 0))
            elif 360<conc[i]<=649:
                aqi = int(round((200 - 151) * (conc[i] - 361) / (649 - 361) + 151, 0))
            elif 649<conc[i]<=1249:
                aqi = int(round((300 - 201) * (conc[i] - 650) / (1249 - 650) + 201, 0))
            elif 1249<conc[i]<=1649:
                aqi = int(round((400 - 301) * (conc[i] - 1250) / (1649 - 1250) + 301, 0))
            else:
                aqi = int(round((500 - 401) * (conc[i] - 1650) / (2049 - 1650) + 401, 0))
        except:
            aqi = np.nan
        c_list.append(aqi)
    return c_list


### AQI subindex calculator for CO
def aqi_easy_10(df, column_index): # CO
    conc = [j for i in df.iloc[:, column_index:(column_index+1)].values.tolist() for j in i]
    c_list = []
    aqi = 0
    c = 0
    for i in range(len(conc)):
        try:
            if conc[i]<=4.4:
                aqi = int(round((50 * conc[i]) / 4.4, 0))
            elif 4.4<conc[i]<=9.4:
                aqi = int(round((100 - 51) * (conc[i] - 4.5) / (9.4 - 4.5) + 51, 0))
            elif 9.4<conc[i]<=12.4:
                aqi = int(round((150 - 101) * (conc[i] - 9.5) / (12.4 - 9.5) + 101, 0))
            elif 12.4<conc[i]<=15.4:
                aqi = int(round((200 - 151) * (conc[i] - 12.5) / (15.4 - 12.5) + 151, 0))
            elif 15.4<conc[i]<=30.4:
                aqi = int(round((300 - 201) * (conc[i] - 15.5) / (30.4 - 15.5) + 201, 0))
            elif 30.4<conc[i]<=40.4:
                aqi = int(round((400 - 301) * (conc[i] - 30.5) / (40.4 - 30.5) + 301, 0))
            else:
                aqi = int(round((500 - 401) * (conc[i] - 40.5) / (50.4 - 40.5) + 401, 0))
        except:
            aqi = np.nan
        c_list.append(aqi)
    return c_list

### AQI subindex calculator for PM2.5
def aqi_easy_6001(df, column_index): # PM2.5
    conc = [j for i in df.iloc[:, column_index:(column_index+1)].values.tolist() for j in i]
    c_list = []
    aqi = 0
    c = 0
    for i in range(len(conc)):
        try:
            if conc[i]<=12.0:
                aqi = int(round((50 * conc[i]) / 12.0, 0))
            elif 12.0<conc[i]<=35.4:
                aqi = int(round((100 - 51) * (conc[i] - 12.1) / (35.4 - 12.1) + 51, 0))
            elif 35.4<conc[i]<=55.4:
                aqi = int(round((150 - 101) * (conc[i] - 35.5) / (55.4 - 35.5) + 101, 0))
            elif 55.4<conc[i]<=150.4:
                aqi = int(round((200 - 151) * (conc[i] - 55.5) / (150.4 - 55.5) + 151, 0))
            elif 150.4<conc[i]<=250.4:
                aqi = int(round((300 - 201) * (conc[i] - 150.5) / (250.4 - 150.5) + 201, 0))
            elif 250.4<conc[i]<=350.4:
                aqi = int(round((400 - 301) * (conc[i] - 250.5) / (350.4 - 250.5) + 301, 0))
            else:
                aqi = int(round((500 - 401) * (conc[i] - 350.5) / (500.4 - 350.5) + 401, 0))
        except:
            aqi = np.nan
        c_list.append(aqi)
    return c_list