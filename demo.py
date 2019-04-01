#!/usr/bin/python 
# -*- coding: UTF-8 -*-

class Classisifer:
    def __init__(self):
        pass

    def binanizer(self, value):
        '''
        二值化
        :return:
        '''
        if value == "True":
            return 1
        else:
            return 0

    def category(self, value, dictionary = []):
        '''
        多类别
        :return:
        '''
        dic = {}
        for i in xrange(len(dictionary)):
            dic.setdefault(dictionary[i], i)
        return dic.get(value, None)

    def numeric(self, value):
        '''
        数值型
        :return:
        '''
        return value

    def model(self):
        '''
        分类器
        :return:
        '''
