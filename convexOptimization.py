from mathFunction import MathFunction
from decimal import Decimal
from enum import Enum
from collections import deque
from copy import deepcopy
from math import inf

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
    
    def set_x0_from_deciam_matrix(self , x0: MathFunction.DecimalMatrix):
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
        if self.searchInterval[0] == None:
            self.get_search_interval()
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
        if self.searchInterval[0] == None:
            self.get_search_interval()
        q = Decimal("0.618")
        a = self.searchInterval[0]
        b = self.searchInterval[1]
                        #  A    A1  A2  B
        que = deque([[a , self.s , None , None] , None , None , [b , self.s , None , None]])   # [[] , [] , [] , []]
        while True:
            j = 0
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
        if que[1][3] > que[2][3]:
            res = [que[2][0] , que[2][2] , que[2][3]]
        else:
            res = [que[1][0] , que[1][2] , que[1][3]]
        self.res = res
        return self.res

    def get_search_interval(self):
        self.searchInterval = determine_search_interval(self.function , self.x0 , self.t0 , self.s)

    def solve(self , method=MethodType.goldenSection , maxSteps=1000):
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
#    def __init__(self, function, x0, t0 , epsilonx: float , epsilonf: float):
#        super().__init__(function, x0, t0)
#        #self.strFunction = function
#        #self.rawX0 = x0
#        self.epsilonx = epsilonx
#        self.epsilonf = epsilonf
#        self.res = None
#        self.oneDimensionProblemMethod=MethodType.goldenSection


    def coordinate_descent(self , epsilon=0):
        if epsilon == 0:
            epsilon = self.epsilonx
        dimension = self.function.dimension
        while True:
            a = Decimal(0)
            for i in range(dimension):
                s = [0 for j in range(dimension)]
                s[i] = 1
                self.set_s(s)

                # 使用一维优化问题求解最优步长
                #p = OnedimensionOptimization(self.strFunction , self.x0 , self.t0 , s , self.epsilonx , self.epsilonf)
                _a , _x , _f = super().solve(method=self.oneDimensionProblemMethod)
                a = max(a , _a)
                self.set_x0_from_deciam_matrix(_x)
            if abs(a) <= self.epsilonx:
                break
        self.res = [a , self.x0 , _f]
        return self.res

    def solve(self , method=MethodType.coordinateDescent):
        if method == MethodType.coordinateDescent:
            return self.coordinate_descent()

if __name__ == "__main__":
    # 寻找区间测试
    function = MathFunction(polynomial="3*x1^3 - 8*x1 + 9")
    print(determine_search_interval(function , MathFunction.DecimalMatrix([[1.8]]) , 0.1 , MathFunction.DecimalMatrix([[1]])))

# 黄金分割测试
    print()
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
        [2 , 2.2],
        0.000001,
        0.01,
        0.01
    )
    res = q.solve()
    print(res)
    for i in res:
        print(i)