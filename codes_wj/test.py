import csv

categories = ['界', '门', '纲', '目', '科', '属']
out_names = ['', '', '', '', '', '', '']
out_list = []

with open('data.txt', 'r', encoding='utf-8') as f:
    names = f.read()
    names = eval(names)

index_previous = 0
for item in names:
    if item != '':
        for index, category in enumerate(categories):
            if category == item[-1]:
                #print(item)
                out_names[index] = item

                out_names[index + 1:] = [''] * len(out_names[index + 1:])
                break
                #print(out_names)
            elif index == 5:
                #print(out_names)
                out_names[-1] = item
                out_list.append(out_names.copy())

with open('final_data.txt', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(out_list)

#print(out_list)