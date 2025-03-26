from mathFunction import MathFunction
from mathFunction import transpose
from decimal import Decimal
from enum import Enum
from collections import deque
from copy import deepcopy
from decimal import Decimal, getcontext

print(getcontext().prec)  # 默认输出 28
getcontext().prec = 50  # 设置更高的精度


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

class Problem:
    class Result:
        def __init__(self , item: dict):
            pass
        def __str__(self):
            pass
    def __init__(self , function: str , x0: list , maxStep=1000):
        '''
        function = x1^1 + 3*x2^2 + ...
        x0 = [1,2,3 ...]
        maxStep 最大迭代步长 默认值1000
        '''
        self.function = MathFunction(function)
        self.x0 = MathFunction.DecimalMatrix([[i] for i in x0]) # 转化为列向量
        self.maxStep = maxStep

    def set_x0_from_list(self , x0):
        self.x0 = MathFunction.DecimalMatrix([[i] for i in x0])

    def set_x0_from_decimal_matrix(self , x0: MathFunction.DecimalMatrix):
        self.x0 = x0

class OnedimensionOptimization(Problem):

    def __init__(self, function: MathFunction, x0: list, s: list , epsilonx: float , epsilonf: float , maxStep=1000):
        '''
        function = x1^1 + 3*x2^2 + ...
        x0 = [1,2,3 ...]
        t0 = 0.1 
        s = [1,2,3]   [1]
        epsilonf 必须传入
        epsilonx 必须传入
        '''
        super().__init__(function, x0 , maxStep)

        self.s = MathFunction.DecimalMatrix([[i] for i in s])
        self.searchInterval = [None , None] # 应该为decimal
        self.epsilonf = epsilonf
        self.epsilonx = epsilonx
        self.res = None

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

    def calculate_golden_point(self , queue: deque):
        a = queue[0][0]
        b = queue[3][0]
        q = Decimal("0.618")
        if queue[1] == None: # 点没有被计算
            queue[1] = [b - q*(b-a) , None , None]
        if queue[2] == None:
            queue[2] = [a + q*(b-a) , None , None]
        self.evaluate_point(queue)
    
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
        while True:
            k1 = cal_k1(quadraticPoint)
            k2 = cal_k2(quadraticPoint , k1)
            if k2 == 0:
                break
            def cal_ap(points , k1 , k2) -> list:
                return [Decimal("0.5")*(points[0][0] + points[2][0]-k1/k2) , None , None]
            ap = cal_ap(quadraticPoint , k1 , k2)
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
        self.res = quadraticPoint[1]
        return self.res

    def golden_section(self , a , b):
        '''
        jMax: 满足函数值间隔的最大迭代次数
        '''
        que = deque([[a , None , None] , None , None , [b , None , None]])   # [[] , [] , [] , []]
        while True:
            step = 0
            self.calculate_golden_point(que)
            while True:
                step += 1
                if que[1][2] > que[2][2]: # F1>F2
                    # F1高，舍弃A
                    # A1 , A2 -> A , A1 , A2 =None
                    que[0] = deepcopy(que[1])
                    que[1] = deepcopy(que[2])
                    que[2] = None
                else: # que[1][3] < que[2][3]
                    que[3] = deepcopy(que[2])
                    que[2] = deepcopy(que[1])
                    que[1] = None
                self.calculate_golden_point(que)
                if step > self.maxStep or abs((que[2][2] - que[1][2])) <= self.epsilonf:
                    break
            if abs((que[2][0] - que[1][0])) < self.epsilonx:
                break
            else:
                que[0] = deepcopy(que[1])
                que[1] = None
                que[3] = deepcopy(que[2])
                que[2] = None
        if que[1][2] > que[2][2]:
            res = que[2]
        else:
            res = que[1]
        self.res = res
        return self.res

    def get_search_interval(self):
        return self.determine_search_interval()

    def solve(self , method=MethodType.goldenSection):
        a , b = self.get_search_interval()
        '''
        返回 [a , x , f]
        '''
        if method == MethodType.goldenSection:
            return self.golden_section(a , b)
        elif method == MethodType.quadraticInterpolation:
            return self.quadratic_interpolation(a , b)

    def determine_search_interval(self):
        '''
        function: 函数
        x0: 起点
        t0: 初始步长
        s: 搜索方向
        return [a: Decimal , b: Decimal]
        '''
        step = Decimal(1) # 步长
        from collections import deque
        queue = deque()
        queue.append([Decimal(0) , None , None])
        queue.append([step , None , None])

        while True:
            self.evaluate_point(queue)
            if queue[1][2] > queue[0][2]: # F2 > F1
                step = -step/Decimal(2) # 步长反向并缩小
                queue.pop() # 将原A2从末尾出队
                nextPoint = [queue[0][0]+step , None , None]
                queue.append(nextPoint)
            else:
                step = step * 2
                break


        while True: # 进退法
            nextPoint = [queue[-1][0]+step , None , None]
            if len(queue) == 3:
                queue.popleft()
            queue.append(nextPoint)
            self.evaluate_point(queue)
            step = step * 2
            if len(queue) == 3 and (queue[0][2] >= queue[1][2] and queue[2][2] >= queue[1][2]):
                break

        res = list(queue)[0:3:2]
        res.sort(key=lambda x: x[0])
        return [i[0] for i in res]


class MultidimensionOptimization(OnedimensionOptimization):
    def __init__(self, function, x0, epsilonx, epsilonf , maxStep=1000):
        s = [1] # 防报错用
        super().__init__(function, x0, s, epsilonx, epsilonf , maxStep)
        self.oneDimensionProblemMethod = MethodType.goldenSection

    def coordinate_descent(self):
        dimension = self.function.dimension
        step = 0
        while True:
            a = Decimal(0)
            step = step + 1
            for i in range(dimension):
                s = [0 for _ in range(dimension)]
                s[i] = 1
                self.set_s(s)
                _a , x , f = super().solve(method=self.oneDimensionProblemMethod)
                a = max(a , abs(_a))
                self.set_x0_from_decimal_matrix(x)
            if abs(a) <= self.epsilonx:
                break
            if step >= self.maxStep:
                raise ValueError(f"迭代达到最大步长.最后的优化步a={a}")
        self.res = [self.x0 , f]
        return self.res

    def gradient_descent(self):
        step = 0
        x = self.x0
        f = Decimal(0)
        while True:
            step = step + 1
            g = self.function.evaluate_gradient(x)
            gNorm = g.frobenius_norm()
            if gNorm == 0:
                # 对于凸优化，gnorm为0时，即最优点（凸函数的hessian矩阵处处正定）
                break
            sMatrix = (- g / gNorm)
            self.set_s(sMatrix)
            # 调用父类一维优化
            _ , x , f = super().solve(self.oneDimensionProblemMethod)
            self.set_x0_from_decimal_matrix(x)
            if abs(gNorm) <= self.epsilonx:
                break
            if step > self.maxStep:
                break
        self.res = [x , f]
        return self.res

    def damped_newton(self):
        step = 0
        x = self.x0
        f = Decimal(0)
        while True:
            step += 1
            h = self.function.evaluate_hessian_matrix(x)
            g = self.function.evaluate_gradient(x)
            h.inverse()
            s = - h*g
            self.set_s(s)
            # 一维优化求步长
            a , x , f = super().solve(self.oneDimensionProblemMethod)
            self.set_x0_from_decimal_matrix(x)
            sNorm = s.frobenius_norm()
            if abs(a*sNorm) <= self.epsilonx:
                break
            if step > self.maxStep:
                break
        self.res = [x , f , step]
        return self.res

    def conjugate_direction(self):
        # 适用范围有限
        # 构造ss
        step = 0
        from collections import deque
        ss = deque()
        for i in range(self.function.dimension):
            s = [[0] for _ in range(self.function.dimension)]
            s[i] = [1]
            s = MathFunction.DecimalMatrix(s)
            ss.append(s)
        # ss存储每一轮搜索中各步所用的方向向量
        for i in range(self.function.dimension):
            # 一共进行n轮，n为维数
            x0 = self.x0
            for s in ss:
                self.set_s(s)
                _a , x , _f = super().solve(method=self.oneDimensionProblemMethod)
                self.set_x0_from_decimal_matrix(x)
                step += 1
            # xn 就是 x
            s = x - x0
            ss.popleft()
            ss.append(s)
            self.set_s(s)
            _a , x , f = super().solve(self.oneDimensionProblemMethod)
            self.set_x0_from_decimal_matrix(x)
            step = step + 1
        self.res = [x , f]
        return self.res

    def powell_method(self):
        ss = []
        for i in range(self.function.dimension):
            s = [[0] for _ in range(self.function.dimension)]
            s[i] = [1]
            s = MathFunction.DecimalMatrix(s)
            ss.append(s)
        k = 0
        step = 0
        while True:
            k = k + 1
            # 本轮的起始点x0
            x0 = self.x0
            fList = [self.function.evaluate(x0)]
            for _s in ss:
                self.set_s(_s)
                a , x , f = super().solve(self.oneDimensionProblemMethod)
                fMin = f
                self.set_x0_from_decimal_matrix(x)
                fList.append(f)
                step = step + 1
            s = x - x0
            x3 = 2*x - x0
            f1 = self.function.evaluate(x0)
            # f2 = self.function.evaluate(x)
            # 此时x就是f对应的点
            f2 = fMin
            f3 = self.function.evaluate(x3)
            # 计算delta_m
            deltaM = fList[0] - fList[1]
            m = 0
            for i in range(1 , self.function.dimension + 1):
                temp = fList[i-1] - fList[i]
                if temp > deltaM:
                    deltaM = temp
                    m = i-1
            if f3 < f1 and (f1 - 2*f2 + f3)*(f1-f2-deltaM)**2 < Decimal("0.5")*deltaM*(f1-f3)**2:
                self.set_s(s)
                a , x , fMin = super().solve(self.oneDimensionProblemMethod)
                f2 = fMin
                # 删去本轮迭代中最大变化量方向
                ss.pop(m)
                # s方向从末尾补上（s=xn - x0）
                ss.append(s)
                # 设置优化点为下一轮迭代的起始点
                self.set_x0_from_decimal_matrix(x)
            elif f > f3:
                self.set_x0_from_decimal_matrix(x3)
                f2 = f3
                fMin = f3
            # 计算Dx，即相邻迭代点之间的长度
            dx = (x0 - x).frobenius_norm()
            # 计算df
            df = abs((f1-f2)/f1)
            if dx < self.epsilonx or df < self.epsilonf:
                break
        self.res = [x , fMin , k , step]
        return self.res

    def quasi_newton(self , method=MethodType.dfp):
        x = self.x0
        # 第一步使用负梯度方向搜索,手动
        e = []
        for i in range(self.function.dimension):
            ei = [0 for _ in range(self.function.dimension)]
            ei[i] = 1
            e.append(ei)
        # 单位矩阵
        inversH = MathFunction.DecimalMatrix(e)
        g = self.function.evaluate_gradient(x)
        s = -inversH * g
        self.set_s(s)
        x0 = x
        a , x , f = super().solve(self.oneDimensionProblemMethod)
        g0 = g
        self.set_x0_from_decimal_matrix(x)
        g = self.function.evaluate_gradient(x)
        step = 0
        while True:
            step = step + 1
            g = self.function.evaluate_gradient(x)
            deltaX = x - x0
            deltaG = g - g0
            transposeDeltaX = transpose(deltaX)
            transposeDeltaG = transpose(deltaG)
            if method == MethodType.dfp:
                e = (deltaX*transposeDeltaX)/((transposeDeltaX*deltaG).frobenius_norm()) - (inversH*deltaG*transposeDeltaG*inversH)/((transposeDeltaG*inversH*deltaG).frobenius_norm())
            elif method == MethodType.bfgs:
                tX = transpose(x)
                e = (deltaX * transposeDeltaX + ((transposeDeltaG*inversH*deltaG).frobenius_norm()*deltaX*transposeDeltaX)/((transposeDeltaX*deltaG).frobenius_norm()) - inversH*deltaG*transposeDeltaX - deltaX*transposeDeltaG*inversH)/((tX*deltaG).frobenius_norm())
            inversH = inversH + e
            s = - inversH * g
            self.set_s(s)
            x0 = x
            g0 = g
            f0 = f
            a , x , f = super().solve(self.oneDimensionProblemMethod)
            self.set_x0_from_decimal_matrix(x)
            if abs(a) < self.epsilonx and abs((f-f0)) < self.epsilonf:
                break
        self.res = [x , f , step]
        return self.res

    def solve(self , method=MethodType.coordinateDescent):
        if method == MethodType.coordinateDescent:
            return self.coordinate_descent()
        elif method == MethodType.gradientDescent:
            return self.gradient_descent()
        elif method == MethodType.dampedNewton:
            return self.damped_newton()
        elif method == MethodType.conjugateDirection:
            return self.conjugate_direction()
        elif method == MethodType.powell:
            return self.powell_method()
        elif method == MethodType.dfp:
            return self.quasi_newton(method=MethodType.dfp)
        elif method == MethodType.bfgs:
            return self.quasi_newton(method=MethodType.bfgs)


if __name__ == "__main__":
    print("==========")
    print("这是无约束优化的测试程序.")
    # 寻找区间测试
    print("寻找搜索区间测试")
    function = MathFunction(polynomial="3*x1^3 - 8*x1 + 9")

# 黄金分割测试
    print("黄金分割法测试")
    q = OnedimensionOptimization(
        "x1^2 + x2^2 - 8*x1 - 12*x2 + 52",
        [2 , 2],
        [0.707 , 0.707],
        0.1,
        0.15,
    )
    q.searchInterval = [Decimal(-3) , Decimal(5)]
    print(q.solve(MethodType.goldenSection))

# 二次插值测试 
    print("二次插值测试")
    q.searchInterval = [Decimal(-3) , Decimal(5)]
    print(q.solve(MethodType.quadraticInterpolation))

# 坐标轮换测试
    print("坐标轮换法测试.")
    q = MultidimensionOptimization(
        "4 + 4.5*x1 - 4*x2 + x1^2 + 2*x2^2 - 2*x1*x2 + x1^4 - 2*x1^2*x2",
        [0 , 0],
        0.01,
        0.01
    )
    res = q.solve()
    print(res)


    q = MultidimensionOptimization(
        "x1^2 + 2*x2^2",
        [0 , 0],
        0.01,
        0.01
    )
    print(q.solve(method=MethodType.gradientDescent)[0])
    print(q.res)

# 阻尼牛顿法
    print("阻尼牛顿法测试.")
    q = MultidimensionOptimization(
        "x1^2 - x1*x2 + x2^2 + 2*x1 - 4*x2",
        [2 , 2],
        0.01,
        0.01
    )
    print(q.solve(method=MethodType.dampedNewton)[0])

# 共轭方向法
    print("共轭方向法测试")
    q = MultidimensionOptimization(
        "2*x1^2 + 2*x1*x2 + 2*x2^2",
        [10 , 10],
        0.01,
        0.01
    )
    q.oneDimensionProblemMethod = MethodType.quadraticInterpolation
    res = q.solve(method=MethodType.conjugateDirection)
    print(res)
    print(res[0])

# powell
    print("powell法测试")
    q = MultidimensionOptimization(
        "11*x1^2 + 11*x2^2 + 18*x1*x2 - 100*x1 - 100*x2 + 250",
        [0 , 0],
        0.001,
        0.001
    )
    print(q.solve(method=MethodType.powell))
    print(q.res[0]) 

# dfp
    print("dfp法测试")
    q = MultidimensionOptimization(
        "4*x1^2 + x2^2 - 40*x1 - 12*x2 + 136",
        [8 , 9],
        0.01,
        0.01
    )
    # q.oneDimensionProblemMethod=MethodType.quadraticInterpolation
    print(q.solve(method=MethodType.bfgs))
    print(q.res[0])