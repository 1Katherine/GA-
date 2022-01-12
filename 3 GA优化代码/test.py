#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：3 GA优化代码 
@File ：test.py
@Author ：Yang
@Date ：2022/1/11 17:46 
'''

# precision = 1
# print(precision % 1 == 0)

'''
    新增mutation代码测试
'''
# import numpy as np
# size_pop = 30
# len_chrom = 81
# Lind = np.array([ 6 , 2 , 3,  3  ,6,  6 ,31,  9 , 6,  9])
# cumsum_len_segment = Lind.cumsum()
# print(cumsum_len_segment)
# mask = np.zeros(shape=(size_pop, len_chrom))
# # i = 0\1\2\3...9（变量个数） j = 6\8\11\14\....
# # 对每一个变量的片段染色体计算 gray2rv （一个变量对应 X 的多个列，有的对应2列 有的对应8列）
# probMut = np.array([0.01, 0.1, 0.02, 0.03, 0.05, 0.06, 0.07, 0.8, 0.03, 0.1])
# d = dict(zip(cumsum_len_segment, probMut))
# for i, segment in enumerate(d):
#     if i == 0:
#         # 取初始种群 Chrom 的前6列 （第一个变量）
#         mask[:, :cumsum_len_segment[0]] = np.random.rand(size_pop, cumsum_len_segment[0]) < d[segment]
#     else:
#         # 取 Chorm 的第 cumsum_len_segment[i - 1] 到第 cumsum_len_segment[i]列（其他变量列）
#         # 取每一个变量对应的染色体子片段
#         size = cumsum_len_segment[i] - cumsum_len_segment[i - 1]
#         mask[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]] = np.random.rand(size_pop, size) < d[segment]

'''
    新增：其他抽样方式测试
    已经采样了实数值，将实数值转为对应的二进制矩阵
'''
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from samples import LHS_sample
class GA2x:
    def __init__(self):
        self.size_pop = 30 # 种群个体数为30
        self.rs_sizepop = int(self.size_pop / 2) # 用于随机采样生成的个体
        self.lhs_sizepop = self.size_pop - self.rs_sizepop # 用于其他采样方式生成的个体
        self.lb = np.array([2,6]) # 两个变量，范围和精度 = [2, 5, 1.0],[6, 22, 0.01]
        self.ub = np.array([5,22])
        self.precision = np.array([1.0, 0.01])
        self.n_dim = len(self.lb)

        self.Lind_raw = np.log2((self.ub - self.lb) / self.precision + 1)
        self.Lind = np.ceil(self.Lind_raw).astype(int)
        self.int_mode_ = (self.precision % 1 == 0) & (self.Lind_raw % 1 != 0) # ndarray(10,) = true/false, 精度不为1，self.int_mode_一定为false
        # 只要precision为整数就是true，或者精度为小数但Lind_raw为整数就是true
        self.int_mode = np.any(self.int_mode_) # 有一个参数变量的（精度为1 && 基因数为偶数）则返回true
        # int_mode_为1，我们需要ub_extend来使这个数字等于2**n
        if self.int_mode:
            # 如果 int_mode_ = true，精度为1，更新上界 ub_extend = self.lb + (np.exp2(self.Lind) - 1) * self.precision
            # 如果 int_mode_ = false，精度不为1，不更新上界 ub_extend = self.ub
            self.ub_extend = np.where(self.int_mode_
                                      , self.lb + (np.exp2(self.Lind) - 1) * self.precision
                                      , self.ub)


    def getRealPrecision(self):
        # 该变量需要的二进制编码总数
        self.bitsPower = np.ceil(self.Lind_raw).astype(int)
        self.realPrecision = np.zeros(shape=(len(self.bitsPower)))
        self.diff = np.zeros(shape=(len(self.bitsPower)))
        for i in range(len(self.bitsPower)):
            # 真实精度
            self.realPrecision[i] = (self.ub[i] - self.lb[i] + 1) / 2 ** self.bitsPower[i]
            # 不知道？
            # self.diff[i] = 2 ** self.bitsPower[i] * self.realPrecision[i] - self.ub[i]

    # x值所在的维度
    def GetCodeParameter(self, x, x_dim):
        # x 对应的十进制数
        codenum = (x - self.lb[x_dim]) / self.realPrecision[x_dim]
        # 十进制转二进制
        code = bin(int(codenum))[2:]
        while len(code) < self.bitsPower[x_dim]:
            # 头部插入0
            code = '0' + code
        # print(code)
        return np.array(list(code))

    def getGrayValue(self, graycode, code_dim):
        X = []
        for code in graycode:
            graycode_str = code.tolist()
            charstr = ''
            for c in graycode_str:
                if c == 0.0:
                    charstr = charstr + '0'
                if c == 1.0:
                    charstr = charstr + '1'
            X.append(int(charstr , 2) * self.realPrecision[code_dim] + self.lb[code_dim])
        return np.array(X)


    def samples(self):
        self.bounds = list(zip(self.lb, self.ub))
        l = LHS_sample.LHSample(self.n_dim, self.bounds, self.lhs_sizepop)
        lhsample = l.lhs()
        print('产生的样本lhsample = \n' + str(lhsample))
        for sample in lhsample:
            for dim in range(self.n_dim):
                if self.precision[dim] == 1.0:
                    sample[dim] = round(sample[dim])
        print('根据精度操作样本lhsample = \n' + str(lhsample))
        return lhsample


    def gray2rv(self, gray_code):
        # len_gray_code 为该变量对应的染色体数量
        _, len_gray_code = gray_code.shape
        # 按照列累加(第一列 = 第一列 ， 第二列 = 第一列+第二列 ， 第三列 = 第一列+第二列+第三列....)，除以2取余 = 0 / 1
        b = gray_code.cumsum(axis=1) % 2
        # np.logspace 对数等比数列 start=开始值，stop=结束值，num=元素个数，base=指定对数的底, endpoint=是否包含结束值
        # mask = 0.5**n (n = 1 .... len_gray_code) mask = [0.5      0.25     0.125    0.0625   0.03125  0.015625]
        mask = np.logspace(start=1, stop=len_gray_code, base=0.5, num=len_gray_code)
        # 返回数组，个数为30（种群个数）
        res = (b * mask).sum(axis=1) / mask.sum()
        return res

    # 样本转二进制种群
    def numTo2x(self, samples):
        self.len_Chrom = self.bitsPower.sum()
        self.Chrom = np.zeros(shape=(len(samples), self.len_Chrom))
        self.cumsum_len_segment = self.bitsPower.cumsum()
        for row, lhs in enumerate(samples):
            X = np.zeros(shape=(self.len_Chrom))
            for dim in range(self.n_dim):
                code = self.GetCodeParameter(lhs[dim], dim)
                if dim == 0:
                    X[:self.cumsum_len_segment[dim]] = code
                else:
                    X[self.cumsum_len_segment[dim - 1]:self.cumsum_len_segment[dim]] = code
            self.Chrom[row:row + 1, :] = X
        return self.Chrom

    def test(self):
        self.getRealPrecision()
        lhsample = self.samples()

        self.Chrom = self.numTo2x(lhsample)

        # print(self.Chrom)
        self.mychrom2x()

        '''
            源码中的 二进制种群计算实数值
        '''
    def chrom2x(self):
        Chrom = self.Chrom
        cumsum_len_segment = self.cumsum_len_segment
        X = np.zeros(shape=(self.lhs_sizepop, self.n_dim))
        # i = 0\1\2\3...9（变量个数） j = 6\8\11\14\....
        # 对每一个变量的片段染色体计算 gray2rv （一个变量对应 X 的多个列，有的对应2列 有的对应8列）
        for i, j in enumerate(cumsum_len_segment):
            if i == 0:
                # 取初始种群 Chrom 的前6列 （第一个变量）
                Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
            else:
                # 取 Chorm 的第 cumsum_len_segment[i - 1] 到第 cumsum_len_segment[i]列（其他变量列）
                # 取每一个变量对应的染色体子片段
                Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
            # 将每一个变量的片段染色体的 0-1 之间的实数值放入返回值X的对应列中
            X[:, i] = self.gray2rv(Chrom_temp)

            if self.int_mode:
                X = self.lb + (self.ub_extend - self.lb) * X
                X = np.where(X > self.ub, self.ub, X)
            else:
                X = self.lb + (self.ub - self.lb) * X
            print('X = \n' + str(X))
            return X

    # 我的：二进制种群计算实数值
    def mychrom2x(self):
        Chrom = self.Chrom
        cumsum_len_segment = self.cumsum_len_segment
        X = np.zeros(shape=(self.lhs_sizepop, self.n_dim))
        # i = 0\1\2\3...9（变量个数）
        for i, j in enumerate(cumsum_len_segment):
            if i == 0:
                # 取初始种群 Chrom 的前6列 （第一个变量）
                Chrom_temp = Chrom[:, :cumsum_len_segment[0]]
            else:
                # 取 Chorm 的第 cumsum_len_segment[i - 1] 到第 cumsum_len_segment[i]列（其他变量列）
                # 取每一个变量对应的染色体子片段
                Chrom_temp = Chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
            # 将每一个变量的片段染色体的 0-1 之间的实数值放入返回值X的对应列中
            X[:, i] = self.getGrayValue(Chrom_temp, i)
        print('种群转二进制，二进制转实数值后的种群样本 X = \n' + str(X))

if __name__ == '__main__':
    ga = GA2x()
    ga.test()







class RCGA:
    '''
        初始化
    '''
    def __init__(self):
        self.size_pop = 30 # 种群个体数为30
        self.rs_sizepop = int(self.size_pop / 2) # 用于随机采样生成的个体
        self.lhs_sizepop = self.size_pop - self.rs_sizepop # 用于其他采样方式生成的个体
        self.lb = np.array([2,6]) # 两个变量，范围和精度 = [2, 5, 1.0],[6, 22, 0.01]
        self.ub = np.array([5,22])
        self.precision = np.array([1.0, 0.01])
        self.n_dim = len(self.lb)

    '''
        lhs采样
    '''
    def samples(self):
        self.bounds = list(zip(self.lb, self.ub))
        l = LHS_sample.LHSample(self.n_dim, self.bounds, self.lhs_sizepop)
        lhsample = l.lhs()
        return lhsample
        # print('lhsample = \n' + str(lhsample))


    def chrom2x(self, Chrom):
        X = self.lb + (self.ub - self.lb) * Chrom
        return X


    def reversechrom2x(self, X):
        Chrom = (X - self.lb) / (self.ub - self.lb)
        return Chrom

    def test(self):
        lhsample = self.samples()
        chrom = self.reversechrom2x(lhsample)
        # print('x = \n' + str(x))
        print('chrom = \n' + str(chrom))

        x = self.chrom2x(chrom)
        print('x = \n' + str(x))

        # print(lhsample)
        for i,sam in enumerate(lhsample):
            for n in range(self.n_dim):
                if self.precision[n] == 1.0:
                    sam[n] = round(sam[n])
                if self.precision[n] == 0.01:
                    # sam[n] = round(sam[n], 2)
                    sam[n] = sam[n]
                # print(i,sam[n])