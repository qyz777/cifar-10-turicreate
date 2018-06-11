from scipy.misc import imsave
import numpy as np


# 解压缩，返回解压后的字典
def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dic = pickle.load(fo, encoding='bytes')
    fo.close()
    return dic


# 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。
for j in range(1, 6):
    dataName = "cifar-10-batches-py/data_batch_" + str(j)
    Xtr = unpickle(dataName)
    print(dataName + " is loading...")
    for i in range(0, 10000):
        # Xtr['data']为图片二进制数据
        img = np.reshape(Xtr[b'data'][i], (3, 32, 32))
        # 读取image
        img = img.transpose(1, 2, 0)
        # Xtr['labels']为图片的标签，值范围0-9，本文中，train文件夹需要存在，并与脚本文件在同一目录下。
        picName = 'train/' + str(i) + "_" + str(Xtr[b'labels'][i]) + '.jpg'
        imsave(picName, img)
    print(dataName + " loaded.")

print("test_batch is loading...")

# 生成测试集图片
testXtr = unpickle("cifar-10-batches-py/test_batch")
for i in range(0, 10000):
    img = np.reshape(testXtr[b'data'][i], (3, 32, 32))
    img = img.transpose(1, 2, 0)
    picName = 'test/' + str(i) + "_" + str(testXtr[b'labels'][i]) + '.jpg'
    imsave(picName, img)
print("test_batch loaded.")
