import os
import shutil

train_root = './data/train'
val_root = './data/val'
data_file = os.listdir('./train')

dog_file = list(filter(lambda x: x[:3]=='dog', data_file))
cat_file = list(filter(lambda x: x[:3]=='cat', data_file))

for i in range(len(cat_file)):
    pic_path = './train/' + cat_file[i]
    if i < len(cat_file) * 0.9:
        obj_path = train_root + '/cat/' + cat_file[i]
    else:
        obj_path = val_root + '/cat/' + cat_file[i]
    shutil.move(pic_path, obj_path)

print('cat 完成')

for i in range(len(dog_file)):
    pic_path = './train/' + dog_file[i]
    if i < len(dog_file) * 0.9:
        obj_path = train_root + '/dog/' + dog_file[i]
    else:
        obj_path = val_root + '/dog/' + dog_file[i]
    shutil.move(pic_path, obj_path)
print('dog 完成')

# 执行过程中可能是由于内存不够的原因，因此执行了多次才完成数据的迁移



