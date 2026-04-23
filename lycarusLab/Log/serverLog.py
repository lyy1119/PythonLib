from datetime import datetime
from typing import Optional, Final, Dict
from color import COLOR_CODES

class ServerLog:
    # 颜色配置映射
    COLOR_CODES = COLOR_CODES

    def __init__(
        self,
        timeFormat: str = '%Y-%m-%d %H:%M:%S',
        outputFile: str = '',
        fileOnly: bool = False,
        noColor: bool = False,
        colorSignOnly: bool = False,
        noTime: bool = False,
        colorStyle: Optional[Dict[str, str]] = None
    ) -> None:
        """
        :param timeFormat: 时间戳格式化字符串
        :param outputFile: 输出日志文件名，为空则不保存
        :param fileOnly: 为True时关闭控制台输出
        :param noColor: 关闭颜色显示
        :param colorSignOnly: 仅渲染状态类型标签 [Info] 的颜色
        :param noTime: 不输出时间戳
        :param colorStyle: 自定义各级别的颜色配置
        """
        self.timeFormat = timeFormat
        self.savePath = outputFile
        self.showConsole = not fileOnly
        self.noColor = noColor
        self.colorSignOnly = colorSignOnly
        self.noTime = noTime

        # 默认颜色样式
        self.currentStyle: Dict[str, str] = {
            "info": "white",
            "succ": "green",
            "warn": "yellow",
            "error": "red",
            "fatal": "purple"
        }

        # 覆盖颜色样式
        if colorStyle:
            for key, value in colorStyle.items():
                if key in self.currentStyle:
                    self.currentStyle[key] = value.lower()

    def _get_timestamp(self) -> str:
        """获取当前格式化时间"""
        if self.noTime:
            return ""
        now = datetime.now()
        return f"[{now.strftime(self.timeFormat)}] "

    def _format_output(self, levelTag: str, styleKey: str, text: str) -> str:
        """核心输出格式化逻辑"""
        timeStamp = self._get_timestamp()
        rawTag = f"[{levelTag}]"
        rawMessage = f" {text}"

        # 获取颜色
        colorName = self.currentStyle.get(styleKey, "white")
        colorCode = self.COLOR_CODES.get(colorName, self.COLOR_CODES["white"])
        endCode = self.COLOR_CODES["end"]

        # 构造控制台输出 (时间戳不带颜色)
        if self.noColor:
            consoleStr = f"{timeStamp}{rawTag}{rawMessage}"
        elif self.colorSignOnly:
            consoleStr = f"{timeStamp}{colorCode}{rawTag}{endCode}{rawMessage}"
        else:
            consoleStr = f"{timeStamp}{colorCode}{rawTag}{rawMessage}{endCode}"

        # 写入文件 (纯文本，无颜色控制符)
        if self.savePath:
            fileStr = f"{timeStamp}{rawTag}{rawMessage}\n"
            with open(self.savePath, 'a+', encoding='utf-8') as f:
                f.write(fileStr)

        return consoleStr

    def info(self, text: str) -> None:
        """输出标准信息"""
        output = self._format_output("Info", "info", text)
        if self.showConsole:
            print(output)

    def succ(self, text: str) -> None:
        """输出成功信息"""
        output = self._format_output("Succ", "succ", text)
        if self.showConsole:
            print(output)

    def warn(self, text: str) -> None:
        """输出警告信息"""
        output = self._format_output("Warn", "warn", text)
        if self.showConsole:
            print(output)

    def error(self, text: str) -> None:
        """输出错误信息"""
        output = self._format_output("Error", "error", text)
        if self.showConsole:
            print(output)

    def fatal(self, text: str) -> None:
        """输出严重错误信息"""
        output = self._format_output("Fatal", "fatal", text)
        if self.showConsole:
            print(output)

# --- 使用场景演示 ---
if __name__ == "__main__":
    # 场景 1: 标准长期运行程序日志 (带时间，全行颜色)
    logger = Log(outputFile="server.log")
    logger.info("系统服务已启动")
    logger.succ("数据库连接成功")
    logger.warn("磁盘空间占用超过 80%")
    logger.error("API 请求超时 (Endpoint: /v1/user)")
    logger.fatal("内核内存溢出，进程强制退出")

    # 场景 2: 仅渲染标签颜色，自定义部分等级颜色
    customLogger = Log(
        colorSignOnly=True,
        colorStyle={"warn": "red"}, # 将警告也设为红色
        timeFormat="%H:%M:%S"      # 缩短时间格式
    )
    print("\n--- 自定义配置输出 ---")
    customLogger.warn("这是一条仅标签渲染且颜色被修改的警告")