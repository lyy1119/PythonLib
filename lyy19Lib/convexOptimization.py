from .mathFunction import MathFunction, FractionFunction, AddFunction, LnFunction, transpose
from decimal import Decimal
from enum import Enum
from collections import deque
from copy import deepcopy
from decimal import Decimal, getcontext
from datetime import datetime
from random import random
import numpy as np
from math import sqrt, inf

# getcontext().prec = 50  # 设置更高的精度


class MethodType(Enum):
    # 一维优化
    goldenSection = 1
    quadraticInterpolation = 2
    # 多维优化
    coordinateDescent = 3
    gradientDescent = 4
    dampedNewton = 5
    conjugateDirection = 6
    powell = 7
    dfp = 8
    bfgs = 9
    # 多维约束优化
    stochasticDirectionMethod = 10
    compositeMethod = 11
    penaltyMethodInterior = 12
    penaltyMethodExterior = 13
    penaltyMethodMixed    = 14


outputAccuracy = 4

class Problem:
    class Result:
        def __init__(self , X: MathFunction.DecimalMatrix , F: Decimal , step: int):
            self.step = step
            self.realX = deepcopy(X)
            self.realF = deepcopy(F)
            # self.outputF = self.realF.quantize(Decimal("0.1")**outputAccuracy)
            self.outputF = self.realF
            self.outputX = deepcopy(X)
            # for i in range(len(self.outputX.data)):
                # for j in range(len(self.outputX.data[0])):
                    # self.outputX.data[i][j] = self.outputX.data[i][j].quantize(Decimal("0.1")**outputAccuracy)
        def __str__(self):
            s = ''
            s += "=============================\n"
            s += "优化结果\n"
            s +=f"迭代次数：{self.step}\n"
            s += "X=\n"
            s +=f"{self.outputX}\n"
            s +=f"函数值F={self.outputF}\n"
            s += "=============================\n"
            return s
    def __init__(self , parameter: dict):
        '''
        function = x1^1 + 3*x2^2 + ...
        x0 = [1,2,3 ...]
        maxStep 最大迭代步长 默认值1000
        '''
        try:
            function    = parameter["function"]
            x0          = parameter["x0"]
        except:
            raise ValueError("传参不完整")
        if "maxStep" in parameter.keys():
            maxStep = parameter["maxStep"]
        else:
            maxStep = 100

        if isinstance(function , MathFunction):
            self.function = function
        else:
            self.function = FractionFunction(function)
        if type(x0) != MathFunction.DecimalMatrix:
            self.x0 = MathFunction.DecimalMatrix([[i] for i in x0]) # 转化为列向量
        else:
            self.x0 = x0
        self.maxStep = maxStep
        self.logs = "" # 日志
        self.output = 3
        self.outputIndent = 0

    def add_log_indent(self):
        self.outputIndent += 1

    def reduce_log_indent(self):
        if self.outputIndent >= 1:
            self.outputIndent -= 1

    def set_output_accuracy(self , prec: int):
        self.output = prec

    def clean_logs(self):
        self.logs = ""

    def read_logs(self):
        return deepcopy(self.logs)

    def write_logs(self , s: str):
        s = s.split("\n")
        s =  '\n'.join('> ' * (self.outputIndent) + line for line in s)
        # print(s)
        self.logs = self.logs + s + "\n"

    def set_x0_from_list(self , x0):
        self.x0 = MathFunction.DecimalMatrix([[i] for i in x0])

    def set_x0_from_decimal_matrix(self , x0: MathFunction.DecimalMatrix):
        self.x0 = x0

class OnedimensionOptimization(Problem):
    def __init__(self, parameter: dict):
        '''
        function = x1^1 + 3*x2^2 + ...
        x0 = [1,2,3 ...]
        t0 = 0.1 
        s = [1,2,3]   [1]
        epsilonf 必须传入
        epsilonx 必须传入
        '''
        try:
            epsilonx    = parameter["epsilonx"]
            epsilonf    = parameter["epsilonf"]
            s           = parameter["s"]
        except:
            raise ValueError("传入参数不完整")
        super().__init__(parameter=parameter)

        self.s = MathFunction.DecimalMatrix([[i] for i in s])
        self.searchInterval = [None , None] # 应该为decimal
        self.epsilonf = epsilonf
        self.epsilonx = epsilonx
        self.res = None
        if type(self) == OnedimensionOptimization:
            self.write_logs("======================================")
            self.write_logs("创建一维优化问题对象")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.write_logs(f"时间： {current_time}")
            self.write_logs("---------------------------------")
            self.write_logs("当前参数")
            self.write_logs("函数：")
            self.write_logs(f"{self.function}")
            self.write_logs("初始点：")
            self.write_logs(f"{self.x0}")
            self.write_logs("搜索方向：")
            self.write_logs(f"{self.s}")
            self.write_logs(f"epsilonx = {self.epsilonx}")
            self.write_logs(f"epsilonf = {self.epsilonf}")
            self.write_logs(f"最大迭代步长： {self.maxStep}")
            self.write_logs("======================================\n")

    def set_s(self , s):
        '''
        s支持传DecimalMatrix , 行list
        '''
        if type(s) == MathFunction.DecimalMatrix:
            self.s = s
        else:
            self.s = MathFunction.DecimalMatrix([[i] for i in s])

    def evaluate_function(self , x: MathFunction.DecimalMatrix):
        return self.function.evaluate(x)

    def evaluate_point(self , queue) -> None:
        '''
        更新队列中的信息
        传入一个可迭代对象，collections.deque最佳
        x0: 初始点
        queue结构： [[A , X , F(X)]]  初始化时令X、F(X)=None
        直接修改整个list
        无返回值
        '''
        for i in queue:
            if i[1] == None:
                i[1] = self.x0 + i[0]*self.s # 计算X=X0+AS
            if i[2] == None:
                i[2] = self.evaluate_function(i[1]) # 计算F(X)

    def quadratic_interpolation(self , a: Decimal , b: Decimal):
        # 计算初始点
        a1 = a
        a3 = b
        a2 = (a1+a3)/2
        quadraticPoint = [
            [a1 , None , None], 
            [a2 , None , None],
            [a3 , None , None]
        ]
        def cal_k1(points: list):
            f3 = points[2][2]
            f1 = points[0][2]
            a3 = points[2][0]
            a1 = points[0][0]
            return (f3-f1)/(a3-a1)
        def cal_k2(points: list , k1):
            f2 = points[1][2]
            f1 = points[0][2]
            a2 = points[1][0]
            a1 = points[0][0]
            a3 = points[2][0]
            return ((f2-f1)/(a2-a1)-k1)/(a2-a3)
        self.evaluate_point(quadraticPoint)
        step = 0
        while True:
            step = step + 1
            self.write_logs(f"迭代：{step}")
            self.write_logs(f"A1={quadraticPoint[0][0]}")
            self.write_logs(f"X1=\n{quadraticPoint[0][1]}")
            self.write_logs(f"F1={quadraticPoint[0][2]}")
            self.write_logs(f"A2={quadraticPoint[1][0]}")
            self.write_logs(f"X2=\n{quadraticPoint[1][1]}")
            self.write_logs(f"F2={quadraticPoint[1][2]}")
            self.write_logs(f"A3={quadraticPoint[2][0]}")
            self.write_logs(f"X1=\n{quadraticPoint[2][1]}")
            self.write_logs(f"F1={quadraticPoint[2][2]}")
            k1 = cal_k1(quadraticPoint)
            k2 = cal_k2(quadraticPoint , k1)
            self.write_logs(f"k1= {k1}")
            self.write_logs(f"k2= {k2}")
            if k2 == 0:
                self.write_logs("k2=0,迭代结束.")
                break
            def cal_ap(points , k1 , k2) -> list:
                return [Decimal("0.5")*(points[0][0] + points[2][0]-k1/k2) , None , None]
            ap = cal_ap(quadraticPoint , k1 , k2)
            self.write_logs(f"Ap={ap[0]}")
            if (ap[0] - quadraticPoint[0][0])*(quadraticPoint[2][0] - ap[0]) < 0:
                break
            f2 = quadraticPoint[1][2]
            fp = Decimal(0)
            ef = abs(f2)
            if ef < self.epsilonf:
                ef = Decimal(1)
            # 插入ap
            for i in range(3):
                if quadraticPoint[i][0] > ap[0]:
                    quadraticPoint.insert(i , ap)
                    self.evaluate_point(quadraticPoint)
                    fp = quadraticPoint[i][2]
                    self.write_logs(f"Xp=\n{quadraticPoint[i][1]}")
                    self.write_logs(f"Fp={fp}")
                    break
            # 弹出
            for i in (1,2):
                if quadraticPoint[i-1][2] > quadraticPoint[i][2] and quadraticPoint[i+1][2] > quadraticPoint[i][2]:
                    # 找到高低高
                    quadraticPoint = quadraticPoint[i-1:i+2]
                    break
            ef1 = abs(f2 - fp)
            if ef1/ef < self.epsilonf:
                break
            if step > self.maxStep:
                raise ValueError("优化超过最大步长.")
        self.write_logs("完成：二次插值优化结束")
        res = quadraticPoint[1]
        ResultRes = deepcopy(quadraticPoint[1][1:3:1])
        ResultRes.append(step)
        self.res = self.Result(ResultRes[0] , ResultRes[1] , ResultRes[2])
        return res

    def golden_section(self , a , b):
        '''
        jMax: 满足函数值间隔的最大迭代次数
        '''
        q = (Decimal(5).sqrt() - Decimal(1))/Decimal(2)
        j = 0
        while True:
            j += 1
            a1 = b-q*(b-a); x1 = self.x0 + a1*self.s; f1=self.function.evaluate(x1)
            a2 = a+q*(b-a); x2 = self.x0 + a2*self.s; f2=self.function.evaluate(x2)
            while True:
                if f1 > f2:
                    a = a1; a1=a2; f1=f2
                    a2=a+q*(b-a); x2=self.x0 + a2*self.s; f2=self.function.evaluate(x2)
                else:
                    b=a2; a2=a1; f2=f1
                    a1=b-q*(b-a); x1=self.x0 + a1*self.s; f1=self.function.evaluate(x1)
                j += 1
                if j > 50:
                    break
                try:
                    exitSign = abs((f2-f1)/f1)
                except:
                    exitSign = abs(f2-f1)
                if exitSign <= self.epsilonf:
                    break
            try:
                exitSign = abs((a2 - a1)/a1)
            except:
                exitSign = abs(a2 - a1)
            if exitSign <= self.epsilonx:
                break
            else:
                a=a1; b=a2
        if f1<f2:
            a = a1; f = f1; x = self.x0 + a1*self.s
        else:
            a = a2; f = f2; x = self.x0 + a2*self.s
        self.res = self.Result(x , f , j)
        return [a, x, f]

    def solve(self , method=MethodType.goldenSection):
        self.write_logs("操作：开始一维优化")
        self.write_logs("操作：确定搜索区间")
        self.add_log_indent()
        a , b = self.determine_search_interval()
        self.reduce_log_indent()
        self.write_logs(f"完成：搜索区间为[{a} , {b}]")
        '''
        返回 [a , x , f]
        '''
        if method == MethodType.goldenSection:
            self.write_logs("操作：黄金分割优化")
            return self.golden_section(a , b)
        elif method == MethodType.quadraticInterpolation:
            self.write_logs("操作：二次插值优化")
            return self.quadratic_interpolation(a , b)

    def determine_search_interval(self):
        """
        更稳健的一维搜索初始区间查找器
        用于 powell 或黄金分割等方法前的预处理
        """
        step = Decimal(str(self.epsilonx)) / Decimal("100")
        a1 = Decimal("0")
        a2 = step
        x = self.x0
        f = self.function.evaluate

        f1 = f(x)
        f2 = f(x + a2 * self.s)

        max_iter = 5000
        iter_count = 0
        threshold = Decimal("0")  # 判断下降趋势的容差
        if f1 == f2:
            self.write_logs("f1=f2,移动a2，再加一个步长")
            a2 = a2 + step
            f1 = f(x)
            f2 = f(x+ a2*self.s)

        if f2 < f1 - threshold:
            # 向前推进
            while iter_count < max_iter:
                iter_count += 1
                step *= 2; a2 += step; f1=f2
                f2 = f(x + a2 * self.s)
                if not (f2 < f1 - threshold):  # 如果不再下降
                    break
                else:
                    a1 = a2 - step
            else:
                raise RuntimeError("搜索区间查找超过最大迭代次数")
        else:
            # 向负方向推进
            step = -step
            while iter_count < max_iter:
                iter_count += 1
                a1 += step; f2 = f1;
                f1 = f(x + a1*self.s)
                if not (f1 < f2 - threshold):
                    break
                else:
                    a2 = a1 - step
                    step *= 2  # 扩大搜索范围
            else:
                raise RuntimeError("搜索区间查找超过最大迭代次数")
        return a1, a2


class MultidimensionOptimization(OnedimensionOptimization):
    def __init__(self, parameter: dict):
        parameter["s"] = [1] # 防报错
        super().__init__(parameter=parameter)
        self.oneDimensionProblemMethod = MethodType.goldenSection
        if type(self) == MultidimensionOptimization:
            self.write_logs("======================================")
            self.write_logs("创建多维优化问题对象")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.write_logs(f"时间： {current_time}")
            self.write_logs("---------------------------------")
            self.write_logs("当前参数")
            self.write_logs("函数：")
            self.write_logs(f"{self.function}")
            self.write_logs("初始点：")
            self.write_logs(f"{self.x0}")
            self.write_logs(f"epsilonx = {self.epsilonx}")
            self.write_logs(f"epsilonf = {self.epsilonf}")
            self.write_logs(f"最大迭代步长： {self.maxStep}")
            self.write_logs("======================================\n")

    def coordinate_descent(self):
        dimension = self.function.dimension
        step = 0
        while True:
            a = Decimal(0)
            for i in range(dimension):
                step = step + 1
                s = [0 for _ in range(dimension)]
                s[i] = 1
                self.set_s(s)
                self.write_logs(f"迭代：{step}")
                self.write_logs(f"优化方向：\n{s}")
                self.add_log_indent()
                _a , x , f = super().solve(method=self.oneDimensionProblemMethod)
                self.reduce_log_indent()
                self.write_logs(f"本次迭代优化结果：\n")
                self.write_logs(f"步长a={_a}")
                self.write_logs(f"X*=\n{x}")
                self.write_logs(f"F*={f}")
                a = max(a , abs(_a))
                self.set_x0_from_decimal_matrix(x)
            if abs(a) <= self.epsilonx:
                break
            if step >= self.maxStep:
                raise ValueError(f"迭代达到最大步长.最后的优化步a={a}")
        res = [self.x0 , f , step]
        self.write_logs("完成：坐标轮换法完成")
        self.res = self.Result(res[0] , res[1] , res[2])
        return res

    def gradient_descent(self):
        step = 0
        x = self.x0
        f = Decimal(0)
        while True:
            step = step + 1
            self.write_logs(f"迭代：{step}")
            g = self.function.evaluate_gradient(x)
            self.write_logs(f"梯度向量为：\n{g}")
            gNorm = g.frobenius_norm()
            self.write_logs(f"梯度的模长为：{gNorm}")
            if gNorm == 0:
                # 对于凸优化，gnorm为0时，即最优点（凸函数的hessian矩阵处处正定）
                break
            sMatrix = (- g / gNorm)
            self.write_logs(f"搜索方向为：\n{sMatrix}")
            self.set_s(sMatrix)
            # 调用父类一维优化
            self.add_log_indent()
            a , x , f = super().solve(self.oneDimensionProblemMethod)
            self.reduce_log_indent()
            self.write_logs(f"本次迭代优化结果：")
            self.write_logs(f"a={a}")
            self.write_logs(f"X*=\n{x}")
            self.write_logs(f"f={f}")
            self.set_x0_from_decimal_matrix(x)
            if abs(gNorm) <= self.epsilonx:
                break
            if step > self.maxStep:
                break
        self.write_logs(f"完成：梯度法优化完成.")
        res = [x , f , step]
        self.res = self.Result(res[0] , res[1] , res[2])
        return res

    def damped_newton(self):
        step = 0
        x = self.x0
        f = Decimal(0)
        while True:
            step += 1
            self.write_logs(f"迭代：{step}")
            h = self.function.evaluate_hessian_matrix(x)
            g = self.function.evaluate_gradient(x)
            h.inverse()
            self.write_logs(f"海塞矩阵逆矩阵为：\n{h}")
            self.write_logs(f"梯度矩阵为：\n{g}")
            s = - h*g
            self.write_logs(f"优化方向S=\n{s}")
            self.set_s(s)
            # 一维优化求步长
            self.add_log_indent()
            a , x , f = super().solve(self.oneDimensionProblemMethod)
            self.reduce_log_indent()
            self.write_logs(f"本次迭代优化结果：")
            self.write_logs(f"A={a}")
            self.write_logs(f"X*=\n{x}")
            self.write_logs(f"F*={f}")
            self.set_x0_from_decimal_matrix(x)
            sNorm = s.frobenius_norm()
            if abs(a*sNorm) <= self.epsilonx:
                break
            if step > self.maxStep:
                break
        self.write_logs(f"完成：阻尼牛顿法优化完成")
        res = [x , f , step]
        self.res = self.Result(res[0] , res[1] , res[2])
        return res

    def conjugate_direction(self):
        # 适用范围有限
        # 构造ss
        step = 0
        round = 0
        from collections import deque
        ss = deque()
        for i in range(self.function.dimension):
            s = [[0] for _ in range(self.function.dimension)]
            s[i] = [1]
            s = MathFunction.DecimalMatrix(s)
            ss.append(s)
        # ss存储每一轮搜索中各步所用的方向向量
        for i in range(self.function.dimension):
            round = round + 1
            # 一共进行n轮，n为维数
            x0 = self.x0
            for s in ss:
                self.set_s(s)
                step += 1
                self.write_logs(f"迭代：第{round}轮， 第{step}步")
                self.write_logs(f"优化方向：")
                self.write_logs(f"S=\n{s}")
                self.add_log_indent()
                _a , x , _f = super().solve(method=self.oneDimensionProblemMethod)
                self.set_x0_from_decimal_matrix(x)
                self.reduce_log_indent()
                self.write_logs(f"本次迭代优化结果：")
                self.write_logs(f"A={_a}")
                self.write_logs(f"X*=\n{x}")
                self.write_logs(f"F*={_f}")
            # xn 就是 x
            s = x - x0
            ss.popleft()
            ss.append(s)

            self.set_s(s)
            step = step + 1
            self.write_logs(f"迭代：第{round}轮， 第{step}步")
            self.write_logs(f"优化方向(新的方向)：")
            self.write_logs(f"S=\n{s}")
            self.add_log_indent()
            _a , x , f = super().solve(self.oneDimensionProblemMethod)
            self.reduce_log_indent()
            self.set_x0_from_decimal_matrix(x)
            self.write_logs(f"本次迭代优化结果：")
            self.write_logs(f"A={_a}")
            self.write_logs(f"X*=\n{x}")
            self.write_logs(f"F*={_f}")
        self.write_logs(f"完成：共轭方向法完成.")
        res = [x , f , step]
        self.res = self.Result(res[0] , res[1] , res[2])
        return res

    def powell_method(self):
        # 构造初始优化方向
        ss = []
        for i in range(self.function.dimension):
            s = [[0] for _ in range(self.function.dimension)]
            s[i] = [1]
            s = MathFunction.DecimalMatrix(s)
            ss.append(s)
        round = 0
        step = 0
        # 优化开始
        while True:
            round = round + 1
            # 本轮的起始点x0
            x0 = self.x0
            fList = [self.function.evaluate(x0)]
            # 本轮优化，以优化列表中的方向优化
            for _s in ss:
                self.set_s(_s)
                step = step + 1
                self.write_logs(f"迭代： 第{round}轮 ， 第{step}次")
                self.write_logs(f"优化方向：")
                self.write_logs(f"S={_s}")
                self.add_log_indent()
                a , x , f = super().solve(self.oneDimensionProblemMethod)
                fMin = f
                self.reduce_log_indent()
                self.set_x0_from_decimal_matrix(x)
                fList.append(f)
                self.write_logs(f"本轮优化结果：")
                self.write_logs(f"A={a}")
                self.write_logs(f"X*=\n{x}")
                self.write_logs(f"F*={f}")

            # 计算共轭方向
            s = x - x0
            self.write_logs(f"计算得本轮({round})的共轭方向为S\n{s}")
            # 计算x3
            x3 = 2*x - x0
            self.write_logs(f"X3={x3}")
            f1 = self.function.evaluate(x0)
            # f2 = self.function.evaluate(x)
            # 此时x就是f对应的点
            f2 = fMin
            f3 = self.function.evaluate(x3)
            self.write_logs(f"F1={f1}\nF2={f2}\nF3={f3}")

            # 计算delta_m
            deltaM = fList[0] - fList[1]
            m = 0
            for i in range(1 , self.function.dimension + 1):
                temp = fList[i-1] - fList[i]
                if temp > deltaM:
                    deltaM = temp
                    m = i-1
            self.write_logs(f"计算得：deltaM={deltaM}")
            # 是否替换判据
            if f3 < f1 and (f1 - 2*f2 + f3)*(f1-f2-deltaM)**2 < Decimal("0.5")*deltaM*(f1-f3)**2:
                self.write_logs(f"需要替换方向")
                self.set_s(s)
                self.write_logs(f"新的方向为S:\n{s}")
                self.add_log_indent()
                a , x , fMin = super().solve(self.oneDimensionProblemMethod)
                f2 = fMin
                self.reduce_log_indent()
                self.write_logs(f"新方向的优化结果：")
                self.write_logs(f"A={a}")
                self.write_logs(f"X*={x}")
                self.write_logs(f"F*={fMin}")
                # 删去本轮迭代中最大变化量方向
                self.write_logs(f"删去方向的行数为：{m}")
                ss.pop(m)
                # s方向从末尾补上（s=xn - x0）
                ss.append(s)
                # 设置优化点为下一轮迭代的起始点
                self.set_x0_from_decimal_matrix(x)
            elif f > f3:
                self.write_logs(f"不需要替换方向")
                self.set_x0_from_decimal_matrix(x3)
                f2 = f3
                fMin = f3
            # 计算Dx，即相邻迭代点之间的长度
            dx = (x0 - x).frobenius_norm()
            self.write_logs(f"deltaX={dx}")
            # 计算df
            # df = abs((f1-f2)/f1)
            df = abs((f1-f2))
            self.write_logs(f"deltaF={df}")
            if dx < self.epsilonx or df < self.epsilonf:
                break
        self.write_logs(f"完成：powell法优化完成.")
        res = [x , fMin , step]
        self.res = self.Result(res[0] , res[1] , res[2])
        return res

    def quasi_newton(self , method=MethodType.dfp):
        step = 0
        x = self.x0
        # 第一步使用负梯度方向搜索,手动
        # 生成单位矩阵
        identityMatrix = []
        for i in range(self.function.dimension):
            ei = [0 for _ in range(self.function.dimension)]
            ei[i] = 1
            identityMatrix.append(ei)
        identityMatrix = MathFunction.DecimalMatrix(identityMatrix)
        inversH = deepcopy(identityMatrix)
        g = self.function.evaluate_gradient(x)
        while True:
            s = - (inversH * g)
            deltaG = g
            deltaX = x
            if (transpose(g)*s).frobenius_norm() > 0:
                s = -g
                inversH = deepcopy(identityMatrix)
            # 优化
            f0 = self.function.evaluate(deltaX)
            self.set_x0_from_decimal_matrix(deltaX)
            self.set_s(s)
            a, x, f = super().solve(self.oneDimensionProblemMethod)
            self.write_logs(f"本次迭代搜索结果:")
            self.write_logs(f"A={a}")
            self.write_logs(f"X*=\n{x}")
            self.write_logs(f"F*={f}")
            step = step + 1
            deltaX = x - deltaX
            g = self.function.evaluate_gradient(x)
            deltaG = g - deltaG

            self.write_logs(f"迭代：{step}")
            self.write_logs(f"当前点的梯度向量为:")
            self.write_logs(f"{g}")
            self.write_logs(f"DeltaX=\n{deltaX}")
            self.write_logs(f"DeltaG=\n{deltaG}")

            if g.frobenius_norm() <= self.epsilonx and deltaX.frobenius_norm() <= self.epsilonx:
                break
            if step > self.maxStep:
                break
            if f0 < f:
                break
            transposeDeltaX = transpose(deltaX)
            transposeDeltaG = transpose(deltaG)
            # 计算修正矩阵
            # dfp法
            if method == MethodType.dfp:
                e = (deltaX*transposeDeltaX)/((transposeDeltaX*deltaG).frobenius_norm()) - (inversH*deltaG*transposeDeltaG*inversH)/((transposeDeltaG*inversH*deltaG).frobenius_norm())
            # bfgs法
            elif method == MethodType.bfgs:
                tX = transpose(x)
                e = (deltaX * transposeDeltaX + ((transposeDeltaG*inversH*deltaG).frobenius_norm()*deltaX*transposeDeltaX)/((transposeDeltaX*deltaG).frobenius_norm()) - inversH*deltaG*transposeDeltaX - deltaX*transposeDeltaG*inversH)/((tX*deltaG).frobenius_norm())
            self.write_logs(f"修正矩阵e：")
            self.write_logs(f"{e}")
            # 计算新的h矩阵
            inversH = inversH + e
            self.write_logs(f"拟黑塞矩阵H的逆为")
            self.write_logs(f"{inversH}")
        self.write_logs(f"完成：dfp/bfgs优化完成")
        res = [x , f , step]
        self.res = self.Result(res[0] , res[1] , res[2])
        return res

    def solve(self , method=MethodType.coordinateDescent):
        if method == MethodType.coordinateDescent:
            self.write_logs("操作：坐标轮换法")
            return self.coordinate_descent()
        elif method == MethodType.gradientDescent:
            self.write_logs("操作：梯度法")
            return self.gradient_descent()
        elif method == MethodType.dampedNewton:
            self.write_logs("操作：阻尼牛顿法")
            return self.damped_newton()
        elif method == MethodType.conjugateDirection:
            self.write_logs("操作：共轭方向法")
            return self.conjugate_direction()
        elif method == MethodType.powell:
            self.write_logs("操作：powell法")
            return self.powell_method()
        elif method == MethodType.dfp:
            self.write_logs("操作：dfp法")
            return self.quasi_newton(method=MethodType.dfp)
        elif method == MethodType.bfgs:
            self.write_logs("操作：bfgs法")
            return self.quasi_newton(method=MethodType.bfgs)
        
class ConstraintOptimization(Problem):
    def __init__(self , parameter):
        # x0先输入一个任意值
        try:
            self.epsilonx = parameter["epsilonx"]
            self.epsilonf = parameter["epsilonf"]
            self.upLimit  = parameter["upLimit"]
            self.lowLimit = parameter["lowLimit"]
            gu = parameter["gu"]
            hv = parameter["hv"]
            function = parameter["function"]
        except:
            raise ValueError("传入参数不完整")
        parameter["x0"] = [0] # 初始化用
        super().__init__(parameter=parameter)
        # 存储gu，hv
        if isinstance(function, str):
            self.function = FractionFunction(function)
        self.gu = []
        self.hv = []
        for i in gu:
            if not isinstance(i, MathFunction):
                i = FractionFunction(i)
            i.update_dimension(self.function.dimension)
            self.gu.append(i)
        for i in hv:
            if not isinstance(i, MathFunction):
                i = FractionFunction(i)
            i.update_dimension(self.function.dimension)
            self.hv.append(i)

        if type(self) == ConstraintOptimization:
            self.write_logs("======================================")
            self.write_logs("创建约束多维优化问题对象")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.write_logs(f"时间： {current_time}")
            self.write_logs("---------------------------------")
            self.write_logs("当前参数")
            self.write_logs("函数：")
            self.write_logs(f"{self.function}")
            self.write_logs("不等式约束gu(X)：")
            gustr = ''
            for i in self.gu:
                gustr += f"{i}" + "\n"
            self.write_logs(gustr)
            self.write_logs("等式约束hv(X)：")
            hvstr = ''
            for i in self.hv:
                hvstr += f"{i}" + '\n'
            self.write_logs(hvstr)
            self.write_logs(f"epsilonx = {self.epsilonx}")
            self.write_logs(f"epsilonf = {self.epsilonf}")
            self.write_logs(f"最大迭代步长： {self.maxStep}")
            self.write_logs("======================================\n")

    def in_feasible_domain(self , x: list) -> bool:
        for i in self.gu:
            if i.evaluate(x) > 0:
                return False
        for i in self.hv:
            if i.evaluate(x) > 10**(-8): # 也许要修改？应该不能用0
                return False
        else:
            return True
        
    def gen_init_point(self) -> MathFunction.DecimalMatrix:
        '''
        输出结果为: [x1 , x2 , ... , xn]
        '''
        upLimit  = self.upLimit
        lowLimit = self.lowLimit
        if len(upLimit) != self.function.dimension or len(lowLimit) != self.function.dimension:
            raise ValueError(f"上下限与函数维度不符，上限：{upLimit}，下限：{lowLimit}，函数维度：{self.function.dimension}")
        # 生成点
        result = []
        for i in range(self.function.dimension):
            a = lowLimit[i]
            b = upLimit[i]
            result.append([a + random()*(b-a)])
        result = MathFunction.DecimalMatrix(result)
        return result
    
    def stochasticDirectionMethod(self):
        def gen_direction(dimension: int):
            vec = np.random.normal(0, 1, size=dimension)
            norm = np.linalg.norm(vec)
            return (vec / norm).tolist()
        def generate_descent_direction(dimension: int , x: MathFunction.DecimalMatrix , t: float):
            descentS = None
            f0 = self.function.evaluate(x)
            # 产生dimension*2个随机方向
            for _ in range(dimension*2):
                s = gen_direction(self.function.dimension)
                s = [[i] for i in s]
                s = MathFunction.DecimalMatrix(s)
                s = s / s.frobenius_norm()
                # 计算沿着该方向的下一点
                xNext = x + t*s
                if self.in_feasible_domain(xNext):
                    fNext = self.function.evaluate(xNext)
                    if fNext < f0:
                        f0 = fNext
                        descentS = s
                # else:
                    # continue
            return descentS , f0
        # 生成初始点
        while True:
            initPoint = self.gen_init_point()
            if self.in_feasible_domain(initPoint):
                break
        # 随机方向法主流程
        self.write_logs(f"初始点为：{initPoint}")
        x = initPoint
        t = 1
        f0 = self.function.evaluate(x)
        step = 0
        x0 = x
        while True:
            step += 1
            f00 = f0
            # 产生随机方向
            while True:
                direction , f0 = generate_descent_direction(self.function.dimension , x , t)
                if direction == None:
                    t = 0.7*t
                else:
                    break
                if t < self.epsilonx:
                    break
                # 生成下降方向的终止条件：生成了下降方向或者步长小于epsilonx
            if direction:
                # 沿着下降方向搜索
                x0 = x + t*direction
                f0 = self.function.evaluate(x)
                t = 1.3*t
                while t > self.epsilonx:
                    x = x0 + t*direction
                    if self.in_feasible_domain(x):
                        f1 = self.function.evaluate(x)
                        if f1 < f0:
                            f0 = f1
                            x0 = x
                            t = 1.3*t
                            continue
                    if t > self.epsilonx:
                        t = t * 0.7
            if (not direction) or abs(f00 - f0) < self.epsilonf:
                break
        self.res = self.Result(x0 , f0 , step)
        return self.res

    def compositeMethod(self):
        def generate_init_composite(dimension: int):
            # 生成2*dimension个点
            result = [] # [x , f(x)]
            for i in range(dimension):
                # 每个维度生成两个可行点
                while len(result) < i*2:
                    while True:
                        newX = self.gen_init_point()
                        if self.in_feasible_domain(newX):
                            break
                    f = self.function.evaluate(newX)
                    result.append([newX , f])
            return result
        points = generate_init_composite(self.function.dimension)
        def cal_center_point(points: list) -> MathFunction.DecimalMatrix:
            length = len(points)
            result = points[0][0]
            for i in range(1,length):
                result = result + points[i][0]
            result = result / length
            return result
        step = 0
        while True:
            step += 1
            # 复合形法主流程
            points.sort(key=lambda x: x[1]) # 升序
            xc = cal_center_point(points)
            fc = self.function.evaluate(xc)
            df = 0
            for i in points:
                df = df + (fc - i[1])**2
            df = (df/2/self.function.dimension).sqrt()
            if df < self.epsilonf:
                break
            else:
                # 计算去掉最高点形成的形心
                xh , fh = points.pop()
                xc = cal_center_point(points)
                if self.in_feasible_domain(xc):
                    reflect = Decimal("1.3")
                    index = -1
                    while True:
                        # 计算反射点
                        xr = xc + reflect*(xc - xh)
                        if self.in_feasible_domain(xr):
                            fr = self.function.evaluate(xr)
                            if fr < fh:
                                points.append([xr , fr])
                                break
                            else:
                                if reflect < self.epsilonx:
                                    xh = points[index]
                                    index -= 1
                                else:
                                    reflect = reflect / 2
                        else:
                            reflect = reflect / 2
                else: # xc不在可行域
                    # 重置上下界
                    xl = points[0]
                    for i in range(self.function.dimension):
                        self.upLimit[i] = max(xl.data[i][0] , xc.data[i][0])
                        self.lowLimit[i] = min(xl.data[i][0] , xc.data[i][0])
                    points = generate_init_composite(self.function.dimension)
        x = xc
        f = self.function.evaluate(x)
        self.res = self.Result(x , f , step)
        return self.res
    
    def penalty_interior(self , r , c=0.6, multiDimensionOptimizationMethod=MethodType.powell, initPoint=None):
        self.write_logs(f"r={r}")
        def create_penalty_function(fun: FractionFunction, r):
            result = AddFunction(fun)
            r = Decimal(str(r))
            for i in self.gu:
                result = result - LnFunction(-i)*r
            return result
        c = Decimal(str(c))
        r = Decimal(str(r))
        if initPoint:
            initPoint = [[i] for i in initPoint]
            x = MathFunction.DecimalMatrix(initPoint)
        else:
            while True:
                x = self.gen_init_point()
                if self.in_feasible_domain(x):
                    break
        # 计算初始
        penaltyFun = create_penalty_function(self.function, r)
        r = r / c
        fmin = penaltyFun.evaluate(x)
        step = 0
        self.write_logs(f"initPoint x=\n{x}, penaltyFunction={fmin}")
        while True:
            step += 1
            r = c*r
            x0 = x
            f0 = fmin
            penaltyFun = create_penalty_function(self.function, r)
            p = MultidimensionOptimization({
                "function"  : penaltyFun,
                "x0"        : x,
                "epsilonx"  : self.epsilonx,
                "epsilonf"  : self.epsilonf
                })
            x, fmin, _ = p.solve(multiDimensionOptimizationMethod)
            try:
                df = abs((f0 - fmin)/f0)
            except:
                df = abs((f0 - fmin))
            dx = (x0 - x).frobenius_norm()
            self.logs += p.logs
            self.write_logs(f"step={step} , fmin={fmin} , x=\n{x}")
            if (df <= Decimal(str(0.00001)) or dx <= Decimal(str(0.001))):
                break
        fmin = self.function.evaluate(x)
        self.write_logs(f"fmin= {fmin}")
        self.res = self.Result(x , fmin , step)
        return self.res

    def penalty_exterior(self, r , c=8):
        class penalty_function(MathFunction):
            def __init__(self, function: FractionFunction, gu: list, hv: list):
                self.function = function
                self.gu = gu
                self.hv = hv
                self.r = 0
                dimension = self.function.dimension
                for i in self.gu:
                    dimension = max(i.dimension, dimension)
                for i in self.hv:
                    dimension = max(i.dimension, dimension)
                self.function.dimension = dimension
                for i in self.gu:
                    i.dimension = dimension
                for i in self.hv:
                    i.dimension = dimension
                self.dimension = dimension

            def set_r(self , r):
                if type(r) != Decimal:
                    r = Decimal(str(r))
                self.r = r

            def evaluate(self, x: MathFunction.DecimalMatrix):
                f = self.function.evaluate(x)
                g = Decimal(0)
                for i in self.gu:
                    g += (max(0 , i.evaluate(x)))**2
                g = g * self.r
                h = Decimal(0)
                for i in self.hv:
                    h += i.evaluate(x)**2
                h = h * self.r
                return f + g + h
            def __str__(self):
                s = ""
                s = s + str(self.function)
                for i in self.gu:
                    s = s + "r*max{0, " + str(i) + "}"
                for i in self.hv:
                    s = s + "r*[" + str(i) + "]^2"
                return s
        def create_penalty_function(r):
            result = self.function
            for i in self.gu:
                result = result + r*(i*i)
            for i in self.hv:
                result = result + r*(i*i)
            return result
        #  外点法求最优解
        x = self.gen_init_point()
        r = r / c
        step = 0
        while True:
            step += 1
            r = r * c
            x0 = x
            penaltyFun = create_penalty_function(r)
            # penaltyFun = penalty_function(self.function, self.gu, self.hv)
            p = MultidimensionOptimization({
                "function"  : penaltyFun,
                "x0"        : x0,
                "epsilonx"  : self.epsilonx,
                "epsilonf"  : self.epsilonf
                })
            p.solve(MethodType.bfgs) # 目前只能用powell法，因为有max函数，目前无法求解其梯度或海塞矩阵
            x = p.res.realX
            f = p.res.realF
            q = Decimal(-inf)
            for i in self.gu:
                q = max(q, i.evaluate(x))
            for i in self.hv:
                q = max(q, i.evaluate(x))
            dx = (x-x0).frobenius_norm()
            if q < 0.001 or (dx <= 0.00001 and r > 10**8):
                break
        self.res = self.Result(x , f , step)

    def solve(self , method: MethodType , *args):
        if method == MethodType.stochasticDirectionMethod:
            return self.stochasticDirectionMethod()
        elif method == MethodType.compositeMethod:
            return self.compositeMethod()
        elif method == MethodType.penaltyMethodInterior:
            return self.penalty_interior(*args)
        elif method == MethodType.penaltyMethodExterior:
            return self.penalty_exterior(*args)
        
class MultiTargetConstraintOptimization(ConstraintOptimization):
    class Result(Problem.Result):
        def __init__(self, X, F, step):
            self.targetFunctionValue = []
            super().__init__(X, F, step)

        def __str__(self):
            s = ''
            s += "=============================\n"
            s += "优化结果\n"
            s +=f"迭代次数：{self.step}\n"
            s += "X=\n"
            s +=f"{self.outputX}\n"
            s +=f"线性加权函数值F={self.outputF}\n"
            for index, value in enumerate(self.targetFunctionValue):
                s +=f"第{index}个目标函数的值为：{value}\n"
            s += "=============================\n"
            return s

        def set_target_function_value(self, value: list):
            self.targetFunctionValue = deepcopy(value)

    def __init__(self, parameter: dict):
        """
        多目标约束优化, 目前只支持线性加权
        function = [[f1(X), omega1], [f2(X), omega2] ...]
        omega为权重
        """
        try:
            function = parameter["function"]
        except:
            raise ValueError("传入参数不完整")
        self.targetFunction = function
        mergeFunction = None
        for i in function:
            f, w = i[0], i[1]
            if isinstance(f, FractionFunction):
                pass
            else:
                f = FractionFunction(f)
            if mergeFunction:
                mergeFunction = mergeFunction + f*FractionFunction(str(w))
            else:
                mergeFunction = f*FractionFunction(str(w))
        parameter["function"] = mergeFunction
        super().__init__(parameter)
        if type(self) == MultiTargetConstraintOptimization:
            self.write_logs("======================================")
            self.write_logs("创建多目标约束多维优化问题对象（线性加权）")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.write_logs(f"时间： {current_time}")
            self.write_logs("---------------------------------")
            self.write_logs("当前参数")
            self.write_logs("函数：")
            for index, f in enumerate(self.targetFunction):
                self.write_logs(f"目标{index+1}：{f}")
            self.write_logs(f"{self.function}")
            self.write_logs("不等式约束gu(X)：")
            gustr = ''
            for i in self.gu:
                gustr += f"{i}" + "\n"
            self.write_logs(gustr)
            self.write_logs("等式约束hv(X)：")
            hvstr = ''
            for i in self.hv:
                hvstr += f"{i}" + '\n'
            self.write_logs(hvstr)
            self.write_logs(f"epsilonx = {self.epsilonx}")
            self.write_logs(f"epsilonf = {self.epsilonf}")
            self.write_logs(f"最大迭代步长： {self.maxStep}")
            self.write_logs("======================================\n")

    def solve(self, method, *args):
        super().solve(method, *args)
        x = self.res.realX
        f = self.res.realF
        step = self.res.step
        newResult = self.Result(x, f, step)
        targetFunctionValue = []
        for i in self.targetFunction:
            targetFunctionValue.append(i[0].evaluate(x))
        newResult.set_target_function_value(targetFunctionValue)
        self.res = newResult
        return newResult