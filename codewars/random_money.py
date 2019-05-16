# encoding: utf-8
import random

lst = {
    'zhangwei': random.uniform(1, 10),
    'lili': random.uniform(1, 10),
    'xiaoming': random.uniform(1, 10),
    'wanggang': random.uniform(1, 10)
}
print('who is the most lucky: ', {v: k for k, v in lst.items()}[max(lst.values())])
