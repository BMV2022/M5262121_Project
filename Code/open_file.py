import os
import numpy as np
import csv

data_k    = []
data_li = ['RecordID','Age','Gender','Height','ICUType','Weight']
def open_fil(path_name):
    i = 0
    data = []
    try:
        while True:
            file_name = os.listdir(path_name)[i]
            file_name1 = file_name.split('.')
            file_name1 = file_name1[:-1]
            if not len(file_name1) == 1:
                file_name1 = '.'.join(file_name1)
            else:
                file_name1 = file_name1[0]
            file_name_path = path_name + file_name
            data.append([file_name1, file_name_path])
            i += 1
    except:
        return data

def du_data1(path_name):
    data_zong = {}
    f = open(path_name, "r",encoding='utf-8')
    st = f.read()
    f.close()
    st_list = st.split('\n')
    for i in st_list:
        data = i.split(',')
        if not len(data)==3:
            continue
        if data[0] == 'Time':
            continue
        if not data[1] in data_k:
            data_k.append(data[1])
        try:
            data_zong[data[1]]
        except:
            data_zong[data[1]] = [float(data[2])]
        else:
            data_zong[data[1]] = data_zong[data[1]] + [float(data[2])]

    return data_zong
def du_data2(path_name):
    data_zong = {}
    f = open(path_name, "r",encoding='utf-8')
    st = f.read()
    f.close()
    st_list = st.split('\n')
    for i in st_list:
        data = i.split(',')
        if not len(data)==6:
            continue
        if data[0] == 'RecordID':
            continue
        data_zong[float(data[0])] = int(data[-1])


    return data_zong
def main1(pat1,pat2,pat3):
    data_hend = []
    data_txt = []
    data_list = []
    # data_a = open_fil("set-a\\")
    k_sw = du_data2(pat3)
    data_a = open_fil(pat1)
    n = 0
    for i in data_a:
        # n +=1
        # if n==3:
        #     break
        data_zong=du_data1(i[1])
        data_list.append(data_zong)
    # print(data_list)
    for i in data_list:
        da = []
        for j in data_k:
            if j in data_li:
                da.append(i[j][0])
                if not j in data_hend:
                    data_hend.append(j)
                if 'RecordID'==j:
                    t = k_sw[i[j][0]]
                    da.append(t)
                    if not 'lab' in data_hend:
                        data_hend.append('lab')
            else:

                d_jun = ''
                d_fang = ''
                d_da = ''
                d_xiao = ''
                d_chu = ''
                d_dang = ''
                d1 = j+'_jun'
                d2 = j+'_fang'
                d3 = j+'_da'
                d4 = j+'_xiao'
                d5 = j+'_chu'
                d6 = j+'_dang'
                if not d1 in data_hend:
                    data_hend.append(d1)
                if not d2 in data_hend:
                    data_hend.append(d2)
                if not d3 in data_hend:
                    data_hend.append(d3)
                if not d4 in data_hend:
                    data_hend.append(d4)
                if not d5 in data_hend:
                    data_hend.append(d5)
                if not d6 in data_hend:
                    data_hend.append(d6)
                try:
                    i[j]
                except:
                    da.append(d_jun)
                    da.append(d_fang)
                    da.append(d_da)
                    da.append(d_xiao)
                    da.append(d_chu)
                    da.append(d_dang)
                else:
                    if len(i[j])==1:
                        d_jun = i[j][0]
                        d_fang = i[j][0]
                        d_da = i[j][0]
                        d_xiao = i[j][0]
                        d_chu = i[j][0]
                        d_dang = i[j][0]
                    else:
                        d_jun = np.mean(i[j])
                        d_fang = np.var(i[j])
                        d_da = max(i[j])
                        d_xiao = min(i[j])
                        d_chu = i[j][0]
                        d_dang = i[j][-1]

                    da.append(d_jun)
                    da.append(d_fang)
                    da.append(d_da)
                    da.append(d_xiao)
                    da.append(d_chu)
                    da.append(d_dang)
        data_txt.append(da)
    print(data_hend)
    # print(data_txt)
    if not os.path.exists(pat2):
        with open(pat2, "w", newline='') as f:
            headers = data_hend
            f_csv = csv.writer(f)
            f_csv.writerow(headers)

    n = 0
    for i in data_txt:
        n+=1
        print(n)
        with open(pat2, 'a', newline='') as fi:
            write = csv.writer(fi)
            write.writerow(i)
# main1()
a = [['set-a\\','set_a_data.csv','Outcomes-a.txt'],['set-b\\','set_b_data.csv','Outcomes-b.txt'],['set-c\\','set_c_data.csv','Outcomes-c.txt']]
for i in a:
    main1(i[0],i[1],i[2])



