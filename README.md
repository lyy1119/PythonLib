# PythonLib
自己写自己用的Python库

## 开发计划

|完成|提出时间|内容|
|---|---|---|
|Yes|2025.3.22|完成无约束一维和多维凸优化的库|
|Yes|2025.3.22|编写泛型类：分式|
|Yes|2025.3.22|编写现有函数类的从多项式相乘解析|
|Yes|2025.3.22|重载现有函数类的乘法运算和加法运算|
|Yes|2025.3.22|重命名现有函数类为：多项式函数类，继承泛型分式类和多项式函数类，编写分式多项式函数类。||
|Yes|2025.03.28|编写约束优化代码|
||2025.04.17|重构函数库|
||2025.04.17|重构优化库|
||2025.04.17|优化方法debug|

## 安装

```bash
# 安装特点版本  （@后为tag）
pip install git+https://github.com/lyy1119/PythonLib.git@v0.1.3
# 安装最新版本
pip install git+https://github.com/lyy1119/PythonLib.git@main
```

## 测试

项目自带一个测试文件，为test.py  

## 文档

### 程序框图

**1.进退法确定搜索区间**  

![进退法确定搜索区间.drawio](https://raw.githubusercontent.com/lyy1119/Imgs/main/img/进退法确定搜索区间.drawio.png)