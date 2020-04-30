# nlp_webQA
A qa application, find right answer in passages that relate to the question.

### 问题
来自于从苏剑林博客中看到的一个问题，[原文链接](https://spaces.ac.cn/archives/5409/comment-page-1)

这是CIPS-SOGOU问答比赛的题目，数据集是 “一个问题，多个相关材料”的模式，题目是尝试从多段材料中找到正确问题的答案，答案一般为一个材料中片段。例如：
```
question: 围魏救赵发生在哪个时期
answer: 战国时期
passage1: 历史趣闻 www.lishiquwen.com 分享: [导读] 导读:“围魏救赵”这句成语指避实就虚、袭击敌人后方以迫使进攻之敌撤回的战术。故事发生在战国时期的魏国国都大梁即现在的开封。
passage2: 故事发生在战国时期的魏国国都大梁即现在的开封。因魏国国都在大梁所以魏国也称梁国,其国君魏惠王也称梁惠王。
passage3: 围魏救赵原指战国时齐军用围攻魏国的方法,迫使魏国撤回攻赵部队而使赵国得救。
passage4: 许攸建议袁绍派轻骑攻袭许都,迎接天子讨伐曹操,曹操首尾难顾,必败无疑(嗅到了围魏救赵的气息)
```

博主提供了一个非常好的实现，而这个库则是我自己的尝试，working on...

### 环境

- Platform: Windows
- Python: 3.7
- Tensorflow: 2.1
- CUBA: 10.1
- GPU: GTX1080

### 数据集
数据集：https://pan.baidu.com/s/11C21BAupOpiYWoOx23J7Mg，密码:dh9w
