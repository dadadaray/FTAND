# 引用requests库
from time import sleep
import sys

# 设置输出的最大数量
sys.setrecursionlimit(10000)
import requests
# 引用BeautifulSoup库
from bs4 import BeautifulSoup

list_all = []
for n in range(1,73):
# 获取数据
    if n != 1:
        res_foods = requests.get('https://www.plantplus.cn/frps/jingji' + '?page=' + str(n))
    else:
        res_foods = requests.get('https://www.plantplus.cn/frps/jingji')
    # 解析数据
    bs_foods = BeautifulSoup(res_foods.text,'html.parser')

    # 查找最小父级标签
    list_foods = bs_foods.find_all('div', class_='infodiv')
    #print(list_foods)
    # 创建一个空列表，用于存储信息


    for food in list_foods:
        # 提取第0个父级标签中的<a>标签
        tag_a = food.find('div')
        #print(tag_a)
        tag_a = tag_a.find('table')
        #print(tag_a)
        #tag_a = tag_a.find('tbody')
        #print(tag_a)
        tag_a = tag_a.find_all('tr')
        #print(type(tag_a))
        for i in tag_a[1:]:
            #print(type(i))
            a = i.find_all('td')
            count = 0
            if a != None:
                for j in a[:-1]:

                    print(j)
                    #print(j.string, end='')
                    if count % 3 == 0:
                        list_all.append('中文名：' + j.text)
                    elif count % 3 == 1:
                        list_all.append('拉丁名：' + j.text)
                    else:
                        list_all.append('功用：' + j.text)
                    count += 1

print(list_all)

with open('森林食物数据.txt', 'w') as f:
    for i in list_all:
        f.write(i+'\n')