from typing import Optional, Final, Dict

COLOR_CODES = {
    # 标准色 (High Intensity)
    "white":  "\033[97m",
    "gray":   "\033[90m",
    "black":  "\033[30m",
    "red":    "\033[91m",
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "blue":   "\033[94m",
    "purple": "\033[95m",
    "cyan":   "\033[96m",

    # 256色 进阶扩展 (部分常用色)
    "orange": "\033[38;5;208m",   # 亮橙色 (适合高亮)
    "pink":   "\033[38;5;213m",   # 粉色 (适合特殊标记)
    "teal":   "\033[38;5;37m",    # 蓝绿色
    "gold":   "\033[38;5;220m",   # 金色
    "lime":   "\033[38;5;118m",   # 酸橙绿 (比普通绿更亮)
    "brown":  "\033[38;5;130m",   # 棕色
    "violet": "\033[38;5;129m",   # 深紫色

    # 样式控制
    "bold":      "\033[1m",       # 加粗
    "underline": "\033[4m",       # 下划线
    "reversed":  "\033[7m",       # 反色 (前景色与背景色对调)
    "end":       "\033[0m"        # 重置所有样式
}