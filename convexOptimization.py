from mathFunction import MathFunction
from decimal import Decimal
from enum import Enum
from collections import deque
from copy import deepcopy

def evaluate(function , queue , x0: MathFunction.DecimalMatrix):
    '''
    更新队列中的信息
    传入一个可迭代对象，collections.deque最佳
    x0: 初始点
    queue结构： [[A , S , X , F(X)]]  初始化时令X、F(X)=None
    '''
    for i in queue:
        if i[2] == None:
            i[2] = x0 + i[0]*i[1] # 计算X=X0+AS
        if i[3] == None:
            i[3] = function.evaluate(i[2])["Decimal"] # 计算F(X)

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

class Problem:
    def __init__(self , function: str , x0: list , t0: float):
        '''
        function = x1^1 + 3*x2^2 + ...
        x0 = [1,2,3 ...]
        t0 = 0.1 
        '''
        self.function = MathFunction(function)
        self.x0 = MathFunction.DecimalMatrix([[i] for i in x0]) # 转化为列向量
        self.t0 = Decimal(t0)

    def set_x0_from_list(self , x0):
        self.x0 = MathFunction.DecimalMatrix([[i] for i in x0])
    
    def set_x0_from_decimal_matrix(self , x0: MathFunction.DecimalMatrix):
        self.x0 = x0

class OnedimensionOptimization(Problem):

    def __init__(self, function, x0, t0 , s: list , epsilonx: float , epsilonf: float):
        '''
        function = x1^1 + 3*x2^2 + ...
        x0 = [1,2,3 ...]
        t0 = 0.1 
        s = [1,2,3]   [1]
        epsilonf 必须传入
        epsilonx 必须传入
        '''
        super().__init__(function, x0, t0)

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

    def calculate_golden_point(self , queue: deque):
        a = queue[0][0]
        b = queue[3][0]
        s = queue[0][1]
        q = Decimal("0.618")
        if queue[1] == None: # 点没有被计算
            queue[1] = [b - q*(b-a) , s , None , None]
        if queue[2] == None:
            queue[2] = [a + q*(b-a) , s , None , None]
        evaluate(self.function , queue , self.x0)
    
    def quadratic_interpolation(self):
        # 计算初始点
        a1 = self.searchInterval[0]
        a3 = self.searchInterval[1]
        a2 = (a1+a3)/2
        quadraticPoint = [
            [a1 , self.s , None , None], 
            [a2 , self.s , None , None],
            [a3 , self.s , None , None]
        ]
        def cal_k1(points: list):
            f3 = points[2][3]
            f1 = points[0][3]
            a3 = points[2][0]
            a1 = points[0][0]
            return (f3-f1)/(a3-a1)
        def cal_k2(points: list , k1):
            f2 = points[1][3]
            f1 = points[0][3]
            a2 = points[1][0]
            a1 = points[0][0]
            a3 = points[2][0]
            return ((f2-f1)/(a2-a1)-k1)/(a2-a3)
        evaluate(self.function , quadraticPoint , self.x0)
        while True:
            k1 = cal_k1(quadraticPoint)
            k2 = cal_k2(quadraticPoint , k1)
            if k2 == 0:
                break
            def cal_ap(points , k1 , k2) -> list:
                return [Decimal("0.5")*(points[0][0] + points[2][0]-k1/k2) , self.s , None , None]
            ap = cal_ap(quadraticPoint , k1 , k2)
            if (ap[0] - quadraticPoint[0][0])*(quadraticPoint[2][0] - ap[0]) < 0:
                break
            f2 = quadraticPoint[1][3]
            fp = Decimal(0)
            # 插入ap
            for i in range(3):
                if quadraticPoint[i][0] > ap[0]:
                    quadraticPoint.insert(i , ap)
                    evaluate(self.function , quadraticPoint , self.x0)
                    fp = quadraticPoint[i][3]
                    break
            
            # 弹出
            for i in range(1,3):
                if quadraticPoint[i-1][3] > quadraticPoint[i][3] and quadraticPoint[i+1][3] > quadraticPoint[i][3]:
                    # 找到高低高
                    quadraticPoint = quadraticPoint[i-1:i+2]
            ef = abs(f2)
            ef1 = abs(f2 - fp)
            if ef1/ef < self.epsilonf:
                break
        self.res = [quadraticPoint[1][0] , quadraticPoint[1][2] , quadraticPoint[1][3]]
        return self.res

    def golden_section(self , jMax: int):
        '''
        jMax: 满足函数值间隔的最大迭代次数
        '''
        a = self.searchInterval[0]
        b = self.searchInterval[1]
                        #  A    A1  A2  B
        que = deque([[a , self.s , None , None] , None , None , [b , self.s , None , None]])   # [[] , [] , [] , []]
        while True:
            j = 0
            self.calculate_golden_point(que)
            while True:
                j += 1
                # 计算各点,及其函数值
                self.calculate_golden_point(que)
                if que[1][3] > que[2][3]: # F1>F2
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
                if j > jMax or abs((que[2][3] - que[1][3])/que[1][3]) <= self.epsilonf:
                    break
            if abs((que[2][0] - que[1][0])/que[1][0]) < self.epsilonx:
                break
            else:
                que[0] = deepcopy(que[1])
                que[1] = None
                que[3] = deepcopy(que[2])
                que[2] == None
        if que[1][3] > que[2][3]:
            res = [que[2][0] , que[2][2] , que[2][3]]
        else:
            res = [que[1][0] , que[1][2] , que[1][3]]
        self.res = res
        return self.res

    def get_search_interval(self):
        self.searchInterval = determine_search_interval(self.function , self.x0 , self.t0 , self.s)

    def solve(self , method=MethodType.goldenSection , maxSteps=1000):
        self.get_search_interval()
        '''
        返回 [a , x , f]
        '''
        if method == MethodType.goldenSection:
            return self.golden_section(maxSteps)
        elif method == MethodType.quadraticInterpolation:
            return self.quadratic_interpolation()

def determine_search_interval(function: MathFunction , x0: MathFunction.DecimalMatrix , t0: float , s: MathFunction.DecimalMatrix):
    '''
    function: 函数
    x0: 起点
    t0: 初始步长
    s: 搜索方向
    return {"float" : [a , b] , "Decimal" : [a , b]}
    '''

    step = Decimal(t0) #步长
    from collections import deque
    queue = deque()
    queue.append([0 , s , None , None])
    queue.append([step , s , None , None])

    evaluate(function , queue , x0)
    if queue[1][3] > queue[0][3]: # F2 < F1
        step = -step # 步长反向
        queue.pop() # 将原A2从末尾出队
    else:
        step = step * 2

    while True: # 进退法
        nextPoint = [queue[-1][0]+step , s , None , None]
        if len(queue) == 3:
            queue.popleft()
        queue.append(nextPoint)
        evaluate(function , queue , x0)
        step = step * 2
        if len(queue) == 3 and (queue[0][3] > queue[1][3] and queue[2][3] > queue[1][3]):
            break

    res = list(queue)[0:3:2]
    res.sort()
    return [i[0] for i in res]


class MultidimensionOptimization(OnedimensionOptimization):
    def __init__(self, function, x0, t0, epsilonx, epsilonf):
        s = [0] # 放报错用
        super().__init__(function, x0, t0, s, epsilonx, epsilonf)
        self.oneDimensionProblemMethod = MethodType.goldenSection

    def coordinate_descent(self , epsilon=0 , maxStep=1000):
        if epsilon == 0:
            epsilon = self.epsilonx
        dimension = self.function.dimension
        step = 0
        while True:
            a = Decimal(0)
            for i in range(dimension):
                s = [0 for j in range(dimension)]
                s[i] = 1
                self.set_s(s)
                _a , _x , _f = super().solve(method=self.oneDimensionProblemMethod)
                a = max(a , abs(_a))
                self.set_x0_from_decimal_matrix(_x)
            if step >= maxStep:
                raise ValueError("迭代达到最大步长.")
            if abs(a) <= self.epsilonx:
                break
            step += 1
        self.res = [a , self.x0 , _f]
        return self.res

    def gradient_descent(self , epsilon , maxStep=1000):
        step = 0
        x = self.x0
        f = self.function.evaluate(x)["Decimal"]
        while True:
            g = self.function.evaluate_gradient(x , format="Decimal")
            gNorm = g.frobenius_norm()
            if gNorm == 0:
                # 对于凸优化，gnorm为0时，即最优点（凸函数的hessian矩阵处处正定）
                break
            sMatrix = (- g / gNorm)
            self.set_s(sMatrix)
            # 调用父类一维优化
            _ , x , f = super().solve(self.oneDimensionProblemMethod , maxStep)
            self.set_x0_from_decimal_matrix(x)
            if abs(gNorm) <= epsilon:
                break
            step = step + 1
            if step > maxStep:
                break
        self.res = [x , f , step]
        return self.res

    def damped_newton(self , epsilon , maxStep=1000):
        step = 0
        x = self.x0
        f = self.function.evaluate(x)["Decimal"]
        while True:
            h = self.function.evaluate_hessian_matrix(x)
            g = self.function.evaluate_gradient(x , "Decimal")
            h.inverse()
            s = - h*g
            self.set_s(s)
            # 一维优化求步长
            a , x , f = super().solve(self.oneDimensionProblemMethod , maxStep)
            self.set_x0_from_decimal_matrix(x)
            step += 1
            sNorm = s.frobenius_norm()
            if abs(a*sNorm) <= epsilon:
                break
            if step > maxStep:
                break
        self.res = [x , f , step]
        return self.res

    def conjugate_direction(self , epsilon , maxStep):
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
                _a , x , _f = super().solve(method=self.oneDimensionProblemMethod , maxSteps=maxStep)
                self.set_x0_from_decimal_matrix(x)
                step += 1
            # xn 就是 x
            s = x - x0
            ss.popleft()
            ss.append(s)
            self.set_s(s)
            _a , x , f = super().solve(self.oneDimensionProblemMethod , maxStep)
            self.set_x0_from_decimal_matrix(x)
            step = step + 1
        self.res = [x , f]
        return self.res

    def powell_method(self , epsilon , maxStep):
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
            fList = [self.function.evaluate(x0)["Decimal"]]
            for _s in ss:
                self.set_s(_s)
                a , x , f = super().solve(self.oneDimensionProblemMethod , maxStep)
                fMin = f
                self.set_x0_from_decimal_matrix(x)
                fList.append(f)
                step = step + 1
            s = x - x0
            x3 = 2*x - x0
            f1 = self.function.evaluate(x0)["Decimal"]
            # f2 = self.function.evaluate(x)["Decimal"]
            # 此时x就是f对应的点
            f2 = fMin
            f3 = self.function.evaluate(x3)["Decimal"]
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
                a , x , fMin = super().solve(self.oneDimensionProblemMethod , maxStep)
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

    def solve(self , method=MethodType.coordinateDescent , maxStep=1000):
        if method == MethodType.coordinateDescent:
            return self.coordinate_descent(maxStep=maxStep)
        elif method == MethodType.gradientDescent:
            return self.gradient_descent(self.epsilonx , maxStep)
        elif method == MethodType.dampedNewton:
            return self.damped_newton(self.epsilonx , maxStep)
        elif method == MethodType.conjugateDirection:
            return self.conjugate_direction(self.epsilonx , maxStep)
        elif method == MethodType.powell:
            return self.powell_method(self.epsilonx , maxStep)


if __name__ == "__main__":
    print("==========")
    print("这是无约束优化的测试程序.")
    # 寻找区间测试
    function = MathFunction(polynomial="3*x1^3 - 8*x1 + 9")
    print(determine_search_interval(function , MathFunction.DecimalMatrix([[1.8]]) , 0.1 , MathFunction.DecimalMatrix([[1]])))

# 黄金分割测试
    print("")
    q = OnedimensionOptimization(
        "x1^2 + x2^2 - 8*x1 - 12*x2 + 52",
        [2 , 2],
        0.1,
        [0.707 , 0.707],
        0.1,
        0.15,
    )
    q.searchInterval = [Decimal(-3) , Decimal(5)]
    print(q.solve(MethodType.goldenSection , 50))

# 二次插值测试 
    print()
    q.searchInterval = [Decimal(-3) , Decimal(5)]
    print(q.solve(MethodType.quadraticInterpolation , 50))

# 坐标轮换测试
    q = MultidimensionOptimization(
        "4 + 4.5*x1 - 4*x2 + x1^2 + 2*x2^2 - 2*x1*x2 + x1^4 - 2*x1^2*x2",
        [0 , 0],
        0.0001,
        0.01,
        0.01
    )
    try:
        res = q.solve()
        for i in res:
            print(i)
    except ValueError as e:
        print(e)


    q = MultidimensionOptimization(
        "x1^2 + 2*x2^2",
        [0 , 0],
        0.01,
        0.01,
        0.01
    )
    print(q.solve(method=MethodType.gradientDescent)[1])

# 阻尼牛顿法
    print()
    q = MultidimensionOptimization(
        "x1^2 - x1*x2 + x2^2 + 2*x1 - 4*x2",
        [2 , 2],
        0.01,
        0.01,
        0.01
    )
    print(q.solve(method=MethodType.dampedNewton)[0])

# 共轭方向法
    print()
    q = MultidimensionOptimization(
        "2*x1^2 + 2*x1*x2 + 2*x2^2",
        [10 , 10],
        0.01,
        0.01,
        0.01
    )
    q.oneDimensionProblemMethod = MethodType.quadraticInterpolation
    res = q.solve(method=MethodType.conjugateDirection)

# powell
    q = MultidimensionOptimization(
        "11*x1^2 + 11*x2^2 + 18*x1*x2 - 100*x1 - 100*x2 + 250",
        [0 , 0],
        0.00000001,
        0.001,
        0.001
    )
    print("=====")
    print(q.solve(method=MethodType.powell))
    print(q.res[0])