import turicreate as tc
import os


# 获得label的SArray
def image_label(frame):
    label = []
    data_path = frame['path']
    for i in range(0, len(data_path)):
        path = str(data_path[i])
        if path[-5] is "0":
            label.append("飞机")
        elif path[-5] is "1":
            label.append("汽车")
        elif path[-5] is "2":
            label.append("鸟")
        elif path[-5] is "3":
            label.append("猫")
        elif path[-5] is "4":
            label.append("鹿")
        elif path[-5] is "5":
            label.append("狗")
        elif path[-5] is "6":
            label.append("青蛙")
        elif path[-5] is "7":
            label.append("马")
        elif path[-5] is "8":
            label.append("轮船")
        elif path[-5] is "9":
            label.append("卡车")
    label_array = tc.SArray(data=label, dtype=str)
    return label_array


# 添加label列
def add_label_column(data):
    sa = image_label(data)
    data = data.add_column(data=sa, column_name='label')
    return data


# 保存SFrame
def save_data(data):
    data.save('cifar-10.sframe')
    return


def create_new_data():
    data = tc.image_analysis.load_images('train', with_path=True)
    data = add_label_column(data)
    save_data(data)
    return


def create_test_data():
    data = tc.image_analysis.load_images('test', with_path=True)
    data = add_label_column(data)
    return data


if __name__ == "__main__":
    if os.path.exists('cifar-10.sframe') is False:
        create_new_data()
    sf = tc.load_sframe('cifar-10.sframe')
    model = tc.image_classifier.create(sf, target='label')
    model.save('cifar-10_model')
    model.export_coreml('cifar-10.mlmodel')
    # 测试准确率
    # model = tc.load_model('cifar-10_model')
    # test_sf = create_test_data()
    # predictions = model.predict(test_sf)
    # result = model.evaluate(test_sf)
    # print(result['accuracy'])
