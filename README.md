# EMNLP Homework

北京大学自然语言处理中的经验性方法课程作业存档

## HW1

在不调NLP包的要求下，实现了一个Log-Linear分类器和一个朴素贝叶斯分类器，在SST-5和Yelp两个数据集上进行测试。除了手写分类器之外最大的收获是学会写argument parser(x).

## HW2

把 event trigger detection 任务视为 sequence tagging 任务，完成了基于特征的线性模型和基于预训练词向量(GloVe)的RNN. 在提取特征的时候 `sklearn` 的 `DictVectorizer` 非常好用😊！由于赶DDL导致analysis部分完成得比较粗糙😭.

## HW3

这次作业是第二次作业的延续，完成了整个 event extraction task. 总共分为四个阶段实现：event trigger detection(第二次作业的内容), event trigger classification, argument identification, argument classification. 最困难的部分应该是 argument identification. 在写这次作业的时间里还并行地在准备保研面试所以完成得比较草率，metrics 都比较低，写得尤其心累😓.

## Summary

写了这些作业之后，一方面是对特征工程有了更多的了解，另一方面也训练了代码规范，当然进步空间也还很大啦！完结撒花🥰
