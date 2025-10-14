from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
from tqdm import tqdm

# 设置ChromeDriver路径
chrome_driver_path = '/usr/local/bin/chromedriver'

# 创建ChromeDriver服务
service = Service(chrome_driver_path)

# 创建Chrome选项
options = Options()

# 确保窗口不会弹出
options.add_argument('--headless')

# 初始化WebDriver
driver = webdriver.Chrome(service=service, options=options)

# 创建列表存储所需数据
list_all = []

def click(i_1, i_2):
    for i in tqdm(range(i_1 + 1, i_2 + 1),  desc="Clicking and loading content", ncols=160, colour='blue'):
        button_id = f'taxa_tree_{i}_switch'
        button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, button_id)))
        button.click()
        time.sleep(0.8)  # 等待内容加载
        # 重新获取页面的HTML内容

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')
    return soup

def extract_chinese(text):
    chinese_text = ''.join([char for char in text if '\u4e00' <= char <= '\u9fff'])
    return chinese_text

try:
    # 打开目标网页
    driver.get('http://www.isenlin.cn/species.html')

    # 等待页面加载
    wait = WebDriverWait(driver, 10)

    # 等待动态内容加载完毕，这里假设第一个按钮出现即认为加载完毕
    wait.until(EC.presence_of_element_located((By.ID, 'taxa_tree_1_switch')))
    time.sleep(0.8)  # 可能需要根据页面加载时间调整

    # 获取页面的HTML内容
    page_source = driver.page_source

    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(page_source, 'html.parser')

    # 找到包含目标内容的div
    div_element = soup.find('div', id='taxa_tree')

    if div_element:
        # 在div内查找所有直接子元素为<li>的标签
        li_elements = div_element.find_all('li', recursive=False)
        if li_elements:
            index_1 = len(li_elements)
            for i in range(1, len(li_elements) + 1):
                # 点击对应的按钮以加载内容
                button_id = f'taxa_tree_{i}_switch'
                button = driver.find_element(By.ID, button_id)
                button.click()
                time.sleep(0.8)  # 等待内容加载
                # 重新获取页面的HTML内容
                page_source = driver.page_source

            page_source = BeautifulSoup(page_source, 'html.parser')
            soup = page_source.find('div',id='taxa_tree')
            print('第一页点击成功')
            index_2 = len(soup.find_all('span', class_='node_name'))
            soup = click(index_1, index_2)
            print('第二页点击成功')
            index_3 = len(soup.find_all('span', class_='node_name'))
            soup = click(index_2, index_3)
            print('第三页点击成功')

            index_4 = len(soup.find_all('span', class_='node_name'))
            soup = click(index_3, index_4)
            print('第四页点击成功')

            index_5 = len(soup.find_all('span', class_='node_name'))
            soup = click(index_4, index_5)
            print('第五页点击成功')

            index_6 = len(soup.find_all('span', class_='node_name'))
            soup = click(index_5, index_6)
            print('第六页点击成功')

            all_names = soup.find_all('span', class_='node_name')
            for i in range(0, len(all_names)):
                list_all.append(all_names[i].text)

            # 提取每个字符串中的中文部分
            chinese_parts = [extract_chinese(text) for text in list_all]

            with open('data.txt', 'w', encoding='utf-8') as f:
                f.write(str(chinese_parts))

            print(list_all)

            print(f'index_1:{index_1} index_2{index_2}')
        else:
            print('没有找到<li>元素')
    else:
        print('没有找到ID为taxa_tree的<div>元素')
except Exception as e:
    print(f'发生错误: {e}')
finally:
    # 关闭浏览器
    driver.quit()