file_path = 'eval_results.log'

# 使用with语句自动管理文件资源
lines= []
with open(file_path, 'r') as file:
    for line in file:  # 逐行读取
        lines.append(line.strip())

import re
pattern = re.compile('test_acc\S{2}\s+\d.\d+')
print(re.findall(pattern,lines[0]))
result = re.findall(pattern,lines[0])
a= ','.join(result)
pattern1 = re.compile('\d.\d+')
result1 = re.findall('\d.\d+',a)
print(result1[0])
print(result1)
print(a)
print(lines[0])