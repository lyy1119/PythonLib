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
    goldenSection = 1

class Question:
    def __init__(self , function: str , x0: list , t0: float):
        '''
        function = x1^1 + 3*x2^2 + ...
        x0 = [1,2,3 ...]
        t0 = 0.1 
        '''
        self.function = MathFunction(function)
        self.x0 = MathFunction.DecimalMatrix([[i] for i in x0]) # 转化为列向量
        self.t0 = Decimal(t0)
        pass

class OneDimansionOptimization(Question):

    def __init__(self, function, x0, t0 , s: list , epsilonf: float , epsilonx: float , method: MethodType):
        '''
        function = x1^1 + 3*x2^2 + ...
        x0 = [1,2,3 ...]
        t0 = 0.1 
        s = [1,2,3]   [1]
        epsilonf 必须传入
        epsilonx 必须传入
        method
        '''
        super().__init__(function, x0, t0)

        self.s = MathFunction.DecimalMatrix([[i] for i in s])
        self.searchInterval = [None , None] # 应该为decimal
        self.epsilonf = epsilonf
        self.epsilonx = epsilonx
        self.method = method
        self.res = None

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

    def solve(self , maxSteps: int):
        if self.method == MethodType.goldenSection:
            return self.golden_section(maxSteps)

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

if __name__ == "__main__":
    function = MathFunction(polynomial="3*x1^3 - 8*x1 + 9")
    print(determine_search_interval(function , MathFunction.DecimalMatrix([[1.8]]) , 0.1 , MathFunction.DecimalMatrix([[1]])))

    # 
    print()
    q = OneDimansionOptimization(
        "x1^2 + 2*x1",
        [0],
        0.1,
        [1],
        0.01,
        0.01,
        MethodType.goldenSection
    )
    q.searchInterval = [Decimal(-3) , Decimal(5)]
    print(q.solve(50))
    #q.get_search_interval()
    #print(q.searchInterval)