import pandas as pd
import numpy as np
from openpyxl import load_workbook

data = pd.read_excel('C:/Users/gogll/Desktop/问卷导出数据.xlsx')

列名称 = data.columns.dropna().unique()  # 数据表所有的列名称
search_col = [col for col in 列名称 if 'named' not in col]  # 去除包含named列名称

分类列表 = ['1.企业营收', '2.综合成本', '2.1原材料成本', '2.2人力成本', '2.3煤电油气等能源成本', '2.4房租成本', '2.5税费成本', '2.6物流成本', '3.主要产品订单量',
        '6.企业应收账款累计余额', '7.企业境内投资额', '8.用工人数']
index = ['增长15%以上', '增长5%-15%', '增长5%以内', '持平', '下降5%以内', '下降5%-15%', '下降15%以上']
企业规模 = ['大型', '中型', '小型', '微型', '第一产业', '第二产业', '第三产业']
write = pd.ExcelWriter(r'C:/Users/gogll/Desktop/处理后的数据.xlsx', engine='openpyxl')

for i in range(len(分类列表)):
    search_column = [col for col in data.columns if 分类列表[i] in col]
    data1 = data.groupby(['企业规模：'])[search_column].count().T
    data2 = data.groupby(['企业主营业务：'])[search_column].count().T
    multi_columns = pd.MultiIndex.from_arrays([[分类列表[i]] * len(企业规模), 企业规模])
    frame = pd.DataFrame(index=index, columns=multi_columns, data=np.hstack([data1.values, data2.values]))
    datax = pd.read_excel(r'C:/Users/gogll/Desktop/处理后的数据.xlsx')
    frame.to_excel(write, sheet_name='Sheet1', startrow=datax.shape[0] + 2)
    write.save()
    write.close()

for i in range(3, len(search_col)):
    index = np.sort(data[search_col[i]].dropna().unique())
    #     print(index)
    if len(index) >= 2:
        multi_columns = pd.MultiIndex.from_arrays([[search_col[i]] * len(企业规模), 企业规模])
        dataframe = pd.DataFrame(index=index, columns=multi_columns)
        for j in range(len(index)):
            for k in range(len(企业规模)):
                列名 = '企业规模：' if k <= 3 else '企业主营业务：'
                dataframe.iloc[j, k] = sum((data[search_col[i]] == index[j]) & (data[列名] == columns[k]))
        datax = pd.read_excel(r'C:/Users/gogll/Desktop/处理后的数据.xlsx')
        dataframe.to_excel(write, sheet_name='Sheet1', startrow=datax.shape[0] + 2)
        write.save()
        write.close()