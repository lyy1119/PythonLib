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

### lycarusLab.Log

一个轻量级的 Python 日志库，提供 Metasploit 风格的交互日志和标准化的服务端长期运行日志，支持跨平台 ANSI 颜色渲染。


#### 1. Color 颜色配置

本库内置了丰富的 ANSI 转义码，支持标准 16 色及部分扩展 256 色。

##### 颜色映射表 (COLOR_CODES)

| 键名 (Key) | 终端显示效果 | 终端支持说明 | 建议用途 |
| :--- | :--- | :--- | :--- |
| `white` | 白色 | 标准 16 色，高兼容性 | 默认信息, 辅助文字 |
| `gray` | 灰色 | 标准 16 色 (亮黑) | 弱化显示, 调试信息 |
| `red` | 红色 | 标准 16 色 | 错误 (Error), 失败 (Fail) |
| `green` | 绿色 | 标准 16 色 | 成功 (Succ) |
| `yellow` | 黄色 | 标准 16 色 | 警告 (Warn) |
| `blue` | 蓝色 | 标准 16 色 | 状态 (Stat) |
| `cyan` | 青色 | 标准 16 色 | 询问 (Ques) |
| `purple` | 紫色 | 标准 16 色 | 严重错误 (Fatal) |
| `orange` | 橙色 | 扩展 256 色 (xterm) | 高亮标记 |
| `gold` | 金色 | 扩展 256 色 (xterm) | 特殊提醒 |

> **注意**：在 Linux (Bash/Zsh) 和现代 Windows (Terminal/PowerShell 7+) 中支持全部颜色。写入文件时，程序会自动剔除颜色代码以保证日志文件的纯净。  

> 对于Windows cmd，需要在注册表中 `HKEY_CURRENT_USER\Console` 下创建 `VirtualTerminalLevel` (DWORD) 并设置为1，以支持颜色。


#### 2. HackLog

`HackLog` 类专为工具类软件设计，模拟 **Metasploit** 的符号输出风格（如 `[+]`, `[-]`, `[*]`, `[!]`, `[?]`）。

##### 初始化参数
- `noColor (bool)`: 是否关闭颜色显示。默认 `False`。
- `colorSignOnly (bool)`: 为 `True` 时仅渲染符号（如 `[+]`），为 `False` 时渲染整行。
- `save (str)`: 指定保存的文件路径，若存在则在输出时同步追加。
- `timestamp (bool)`: 是否开启相对时间戳，以 `[秒数]` 风格显示。
- `colorStyle (dict)`: 用于覆盖默认颜色配置（支持部分覆盖）。

##### 示例代码
```python
from lycarusLab.Log import HackLog

# 开启时间戳，并仅渲染符号颜色
logger = HackLog(timestamp=True, colorSignOnly=True)

logger.stat("正在初始化引擎...")
logger.succ("目标连接成功")
logger.warn("检测到异常流量")
target = logger.ques("输入攻击目标:")
```


#### 3. ServerLog (Log 类)

`Log` 类适用于长期运行的后台程序或服务，采用 `[时间] [级别] 内容` 的标准化格式，适配 `tail -f` 监控。

##### 初始化参数
- `timeFormat (str)`: 时间戳格式，默认 `%Y-%m-%d %H:%M:%S`。
- `outputFile (str)`: 日志存储路径，开启后同步写入文件。
- `fileOnly (bool)`: 设为 `True` 则只写文件，不在控制台打印。
- `noTime (bool)`: 设为 `True` 则不输出时间戳。
- `colorSignOnly (bool)`: 仅渲染级别标签（如 `[Info]`）的颜色。

##### 示例代码
```python
from lycarusLab.Log import ServerLog as Log

# 设置输出文件，自定义 Warn 级别显示为红色
logger = Log(
    outputFile="service.log", 
    colorStyle={"warn": "red"}
)

logger.info("服务已启动")
logger.succ("主逻辑加载完成")
logger.error("配置文件解析失败")
logger.fatal("内存溢出，进程终止")
```

##### 核心特性
* **时间戳隔离**：颜色渲染逻辑仅针对标签或正文，不会渲染时间戳，确保长期运行日志的时间可读性。
* **环境自适应**：统一使用标准 ANSI 转义码，不再区分操作系统环境参数。
* **双重输出**：支持控制台显示与文件追加同步进行，文件输出自动过滤 ANSI 字符。
