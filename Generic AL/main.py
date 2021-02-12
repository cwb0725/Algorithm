"""
遗传算法求解最短路径问题
"""

import codecs
import numpy as np
import random
import matplotlib.pyplot as plt

class Optimization:
    def __init__(self, group_size, generic_times, crossover_probability, mutation_probability, data_num):
        # 遗传算法参数
        self.group_size = group_size  # 种群大小
        self.generic_times = generic_times  # 遗传代数
        self.crossover_probability = crossover_probability  # 交叉概率
        self.mutation_probability = mutation_probability  # 变异概率
        # self.gene_length = gene_length  # 基因长度

        # 系统参数
        self.group = []  # 初始的种群（基因序列, 由 initializer() 初始化）
        self.longtitude = None  # 由 initializer() 初始化
        self.latitude = None  # 由 initializer() 初始化
        self.data_num = data_num  # 总数据条数
        self.distance_matrix = np.zeros((self.data_num + 2, self.data_num + 2))  # 距离矩阵（各点之间的距离矩阵, 由 initializer() 初始化）
        self.data_dir = r"./data.txt"

        # 系统初始化
        self.initializer()

    def initializer(self):
        """
        优化器初始化组件
        :return: None
        """
        print("加载坐标数据...")
        data = self.import_data()  # 获取坐标数据

        # 初始化巡航坐标
        self.longtitude = np.hstack(([70.0], data[:, 0], data[:, 2], data[:, 4], data[:, 6], [70.0]))  # 偶数列为经度
        self.latitude = np.hstack(([40.0], data[:, 1], data[:, 3], data[:, 5], data[:, 7], [40.0]))  # 奇数列为纬度

        # 初始化距离矩阵
        # 将坐标角度转换为弧度制数据
        print("初始化距离矩阵...")
        long = self.longtitude * (np.pi / 180)
        lat = self.latitude * (np.pi / 180)

        # 计算每个点之间的距离，并存放于 dis_matrix 矩阵中
        for i in range(102):
            for j in range(102):
                if i == j:  # 自己与自己的距离为 0
                    continue

                # 三维距离公式求解两点间距离
                tmp = np.cos(long[i] - long[j]) * np.cos(lat[i]) * np.cos(lat[j]) + np.sin(lat[i]) * np.sin(lat[j])
                self.distance_matrix[i][j] = 6370 * np.arccos(tmp)

        # 初始化父代种群
        print("初始化父类亲本中...")
        for _i in range(self.group_size):
            # 随机初始化一个种群
            arr = [x for x in range(1, 101)]
            random.shuffle(arr)  # 打乱顺序
            arr.insert(0, 0)
            arr.append(101)

            # 改良圈算法优化亲本
            flag = 1
            while flag > 0:
                flag = 0
                for i in range(0, (self.data_num + 2) - 3):
                    for j in range(i + 2, (self.data_num + 2) - 1):
                        d1 = self.distance_matrix[arr[i]][arr[j]] + self.distance_matrix[arr[i + 1]][arr[j + 1]]
                        d2 = self.distance_matrix[arr[i]][arr[i + 1]] + self.distance_matrix[arr[j]][arr[j + 1]]

                        if d1 < d2:
                            flag = 1
                            arr[(i + 1):(j + 1)] = arr[(i + 1):(j + 1)].copy()[::-1]

            temp = np.zeros((102,))
            for i in range(len(arr)):
                temp[arr[i]] = i + 1

            # 编码
            # temp = temp / 102
            # temp[0] = 0
            # temp[101] = 1

            self.group.append(temp)

        print("初始化完毕!")

    def core(self):
        """
        算法核心组件
        :return: None
        """
        # 复制产生优良父代 gen0
        gen0 = self.group.copy()

        print("寻找全局最优解中...")
        for _x in range(self.generic_times):
            # 交配(交叉)产生子代 gen1
            gen1 = gen0.copy()

            # 奇数位置与偶数位置上的个体基因进行交叉互换（0 - 49）
            for i in range(0, self.group_size, 2):
                # 随机匹配交配个体
                position = np.arange(self.group_size)
                np.random.shuffle(position)

                # 随机生成交叉点
                t = np.random.randint(1, 101)

                # 基因交配
                toil1, toil2 = gen1[position[i]][t:].copy(), gen1[position[i + 1]][t:].copy()
                gen1[position[i]][t:] = toil2
                gen1[position[i + 1]][t:] = toil1

            # 变异产生子代 gen2
            gen2 = None  # 占位符号（后面将其替换）
            hvp = []  # 变异子代下标
            mask_hvp = np.random.rand(self.group_size) < self.mutation_probability  # 获取满足变异的个体(0.1)
            for i in range(self.group_size):
                if mask_hvp[i]:
                    hvp.append(i)

            # 若变异子代为空，则随机生成一个变异子代
            if len(hvp) == 0:
                hvp.append(np.random.randint(self.group_size))

            # 变异产生的新个体
            gen2 = np.array(gen0)[hvp]

            for i in range(len(hvp)):
                # 随机生成三个分段值 [a, b, c]
                # 将 a - b 之间的基因拷贝到 c 后, 0 - a, b - c, c - end, a - b
                div = np.sort(random.sample(range(1, 101), 3))  # 随机生成三个不同的切割整数
                p1 = gen2[i][:div[0]].copy()  # [0, a)
                p2 = gen2[i][div[0]:(div[1] + 1)].copy()  # [a, b + 1) 包含 a, b 在内
                p3 = gen2[i][(div[1] + 1):div[2]].copy()  # [b + 1, c)
                p4 = gen2[i][div[2]:].copy()  # [c, end)

                gen2[i] = np.hstack((p1, p3, p4, p2))  # 变异基因组合

            all = np.vstack((gen0, gen1, gen2))

            # 将编码升序得到每个个体的表现型（巡航路径）
            all_arg = []  # 个体表现型列表
            for v in all:
                all_arg.append(np.argsort(v))  # argsort 得到排序后的下标值

            # 计算个体内距离
            all_distance = []  # 个体的路径长度
            for w in all_arg:
                all_distance.append(self.decode(w))

            optimal = np.argsort(all_distance)  # 排序后排序下标

            # 选择最优个体作为下一代的亲本（M）
            for i in range(self.group_size):
                gen0[i] = all[optimal[i]]

        print("寻找完毕!")
        return np.argsort(gen0[0])  # 最优解巡航路径

    def plot(self, result):
        result_distance = self.decode(result)
        print("最优巡航路径路程为: ", result_distance)

        x = [self.longtitude[i] for i in result]
        y = [self.latitude[i] for i in result]
        plt.figure(figsize=(10, 7))
        plt.title('\nOptimal Course Chart\n', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=14)  # 设置图表标题和标题字号
        plt.xlabel('longtitude', fontsize=14)
        plt.ylabel('latitude', fontsize=14)
        plt.plot(x, y, alpha=0.8, marker='o', mec='r', mfc='w')
        plt.show()

    def store(self, result, file_dir):
        x = [self.longtitude[i] for i in result]
        y = [self.latitude[i] for i in result]

        re = []
        for i in range(len(x)):
            re.append((x[i], y[i]))

        with open(file=file_dir, mode='w', encoding='utf-8') as f:
            f.write("Optimal Course Data\n(longtitude, latitude)\n")
            for i in re:
                f.write("(" + str(i[0]) + ", " + str(i[1]) + ")\n")

    def import_data(self):
        """
        从文件读取数据
        :return: numpy 数组
        """
        f = codecs.open(self.data_dir, mode='r', encoding='utf-8')  # 以 utf-8 方式打开 txt 文件
        data = []  # 存储所有的坐标数据

        line = f.readline()  # 以行的形式进行读取文件
        while line:
            _tmp = []

            for ele in line.split():
                _tmp.append(float(ele))  # 将字符串数据转换为浮点型
            data.append(_tmp)

            line = f.readline()

        f.close()
        return np.array(data)

    def decode(self, gene) -> int:
        """
        解码, 基因 -> 表现型, 飞行的距离
        :param gene: 基因序列 list/numpy array
        :return: 表现型
        """
        distance = 0
        for i in range(len(gene)):
            if i == (len(gene) - 1):
                break
            distance = distance + self.distance_matrix[gene[i]][gene[i + 1]]

        return distance

if __name__ == '__main__':
    # 遗传算法参数
    _group_size = 100  # 种群大小
    _generic_times = 100  # 遗传最大代数
    _crossover_probability = 1  # 交叉概率（保证充分交配, 该参数无效）
    _mutation_probability = 0.1  # 变异概率
    _data_num = 100  # 数据条数

    # group_size, generic_times, crossover_probability, mutation_probability, data_num
    opt = Optimization(_group_size, _generic_times, _crossover_probability, _mutation_probability, _data_num)
    r = opt.core()
    opt.store(r, r"./result/result_txt.txt")
    opt.plot(r)