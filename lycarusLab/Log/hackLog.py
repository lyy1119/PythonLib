import os
import time
from typing import Optional, Final, Dict
from color import COLOR_CODES

class HackLog:
    # 基础 ANSI 颜色映射表
    COLOR_CODES = COLOR_CODES

    def __init__(
        self,
        noColor: bool = False,
        colorSignOnly: bool = False,
        save: Optional[str] = None,
        timestamp: bool = False,
        colorStyle: Optional[Dict[str, str]] = None
    ) -> None:
        self.noColor = noColor
        self.colorSignOnly = colorSignOnly
        self.savePath = save
        self.useTimestamp = timestamp
        self.startTime = time.time()

        # 1. 设置默认颜色配置
        self.currentStyle: Dict[str, str] = {
            "stat": "white",
            "succ": "green",
            "fail": "red",
            "warn": "yellow",
            "ques": "cyan"
        }

        # 2. 尝试从环境变量覆盖 (优先级: 中)
        # 对应的环境变量名如: STAT_COLOR, SUCC_COLOR 等
        for key in self.currentStyle.keys():
            envValue = os.getenv(f"{key.upper()}_COLOR")
            if envValue:
                self.currentStyle[key] = envValue.lower()

        # 3. 尝试从构造函数参数覆盖 (优先级: 高)
        if colorStyle:
            for key, value in colorStyle.items():
                if key in self.currentStyle:
                    self.currentStyle[key] = value.lower()

    def _get_timestamp(self) -> str:
        """获取相对于启动时间的秒数偏移"""
        if not self.useTimestamp:
            return ""
        elapsed = time.time() - self.startTime
        return f"[{elapsed:.2f}] "

    def _format_message(self, symbol: str, styleKey: str, message: str) -> str:
        """核心格式化逻辑，支持全文本渲染或仅符号渲染"""
        ts = self._get_timestamp()
        rawPrefix = f"[{symbol}]"
        rawContent = f" {ts}{message}"

        # 获取颜色代码
        colorName = self.currentStyle.get(styleKey, "white")
        colorCode = self.COLOR_CODES.get(colorName, self.COLOR_CODES["white"])
        endCode = self.COLOR_CODES["end"]

        # 处理带颜色的输出
        if self.noColor:
            coloredOutput = rawPrefix + rawContent
        elif self.colorSignOnly:
            # 仅渲染符号部分: [?] 0.00 message
            coloredOutput = f"{colorCode}{rawPrefix}{endCode}{rawContent}"
        else:
            # 渲染整个语句: [?] 0.00 message
            coloredOutput = f"{colorCode}{rawPrefix}{rawContent}{endCode}"

        # 写入文件 (始终为纯文本)
        if self.savePath:
            with open(self.savePath, "a", encoding="utf-8") as f:
                f.write(rawPrefix + rawContent + "\n")

        return coloredOutput

    def stat(self, message: str) -> None:
        """状态信息"""
        print(self._format_message("*", "stat", message))

    def succ(self, message: str) -> None:
        """成功信息"""
        print(self._format_message("+", "succ", message))

    def fail(self, message: str) -> None:
        """失败信息"""
        print(self._format_message("-", "fail", message))

    def warn(self, message: str) -> None:
        """警告信息"""
        print(self._format_message("!", "warn", message))

    def ques(self, message: str) -> str:
        """询问信息 (返回用户输入)"""
        formattedPrompt = self._format_message("?", "ques", message)
        # 注意：如果渲染整行颜色，input的提示符也会带颜色
        return input(formattedPrompt + " ")

# --- 演示代码 ---
if __name__ == "__main__":
    # 场景 1: 默认设置 (整行渲染颜色)
    print("--- 默认整行渲染 ---")
    log1 = HackLog(timestamp=True)
    log1.stat("正在检查系统环境...")
    log1.succ("环境检查通过")

    # 场景 2: 仅渲染符号 + 自定义部分颜色
    print("\n--- 仅渲染符号 + 自定义颜色 (stat设为blue) ---")
    log2 = HackLog(colorSignOnly=True, colorStyle={"stat": "blue"})
    log2.stat("这是一条仅符号带颜色的蓝色状态信息")
    log2.warn("这是一条仅符号带颜色的警告")

    # 场景 3: 模拟环境变量覆盖 (STAT_COLOR=cyan)
    print("\n--- 模拟环境变量覆盖 (STAT_COLOR=cyan) ---")
    os.environ["STAT_COLOR"] = "CYAN" 
    log3 = HackLog()
    log3.stat("由于环境变量设置，我现在是青色")