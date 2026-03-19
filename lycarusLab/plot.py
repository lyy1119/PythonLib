import matplotlib.pyplot as plt
from contextlib import contextmanager

@contextmanager
def plot_style(title="", xlabel="", ylabel="", drag=True, grid=True):
    """
    自定义 Matplotlib 绘图上下文管理器

    参数:
    :param title:   图表标题
    :param xlabel:  X 轴标签
    :param ylabel:  Y 轴标签
    :param drag:    是否开启图例拖拽功能
    :param grid:    是否显示网格线
    :param figsize: 画布的尺寸 (宽, 高)
    """
    # 【前置步骤】：创建并初始化画布
    plt.figure()

    try:
        # yield 之前的代码在 with 块开始时执行
        yield 
    finally:
        # 【后置步骤】：在 with 块结束后自动执行装饰逻辑
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if grid:
            plt.grid(True, alpha=0.8)

        # 自动获取图例对象并设置
        leg = plt.legend()
        if leg and drag:
            try:
                leg.set_draggable(True)
            except Exception:
                # 防止由于没有设置 label 导致 legend 为 None 时报错
                pass

        plt.tight_layout() # 自动优化布局
        plt.show()