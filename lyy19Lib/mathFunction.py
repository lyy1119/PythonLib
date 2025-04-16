'''README
该库为多项式函数库，不支持三角函数、指数函数、对数函数。使用泰勒展开式代替
对于函数。需要写成 系数乘以关于x1到xn的单项式的和的形式
目前只支持 小数、整数形式的系数 和 整数形式的次数。对应分数系数，使用一定精度的小数代替

函数书写要求：
1.不用写f(X)
2.变量必须要有下标，如x1
3. 1和-1系数可写成如： x1  ， -x1
4.次数为负数时必须使用括号包围，如： x1^(-5)
5.系数和变量及次数之间必须使用 * （星号）连接，如： x1^2*x2^3
6.次数为1是不需要写次数

书写示例：
x1^2 - 8 + 5*x2^7*x1^2 + x1^(-5) + x5
'''
from decimal import Decimal
from .genericClass import Matrix as GenericMatrix
from .genericClass import transpose
from .genericClass import Fraction as GenericFraction
from collections import deque
from copy import deepcopy
from math import inf
from enum import Enum
import math
import re

def gcd(a , b):
    if a < b:
        a , b = b , a
    while b > 0:
        a = a % b
        a , b = b , a
    return a

def divide_by_plus_minus(str) -> list:
    '''
    input: str = "x1+x2+[x1+x2]^2+[x1+[x2+x3]^2]*[x3+4]"
    output: li = [ "x1" , "x2" , "[x1+x2]^2" , "[x1+[x2+x3]^2]*[x3+4]" ]
    '''
    result = [""]
    que = deque(str)
    skipSign = 0 # 记录当前位置是否在括号中
    while que:
        i = que.popleft()
        if i in "[({":
            skipSign += 1
        elif i in "])}":
            skipSign -= 1
        # 当不在括号中且当前字符等于-或+
        if skipSign == 0 and i in "+-" and result[-1]:
            # 是一个新的单项式或多项式
            result.append("")
        # 将内容加到li中
        result[-1] += i
    return result
 
def filter_space(s) -> str:
    result = ""
    for i in s:
        if i != " ":
            result += i
    return result

class MathFunction:
    class DecimalMatrix(GenericMatrix):
        def __init__(self, li, check=True):
            if type(li[0][0]) != Decimal:
                li = [[Decimal(str(j)) for j in i] for i in li]
            super().__init__(li, check)
        
        def __truediv__(self , other):
            if type(other) != Decimal:
                    other = Decimal(other)
            from copy import deepcopy
            newData = deepcopy(self.data)
            for i in range(self.row):
                for j in range(self.col):
                    try:
                        newData[i][j] /= other
                    except ZeroDivisionError:
                        newData[i][j] = Decimal(inf)
                        # newData[i][j] = Decimal(inf)
            return type(self)(newData)

        def inverse(self):
            '''
            不可逆操作，若要保留原矩阵，使用deepcopy
            没有做是否可逆的检查
            返回DecimalMatrix这个类

            '''
            if self.col != self.row:
                raise ValueError("当前矩阵不是方阵，不能求逆矩阵")
            # 通过行交换，保证data[k][k]不为0
            for i in range(self.row): # row = col
                if self.data[i][i] == 0:
                    for j in range(i+1 , self.row):
                        if self.data[j][i] != 0:
                            self.row_transformation(i , j)
                            break
                # else:
                #     continue
            # 逆矩阵
            return self.gauss_jordan_inverse()

        def gauss_jordan_inverse(self):
            # 拓展为增广矩阵
            for i in range(self.row):
                for j in range(self.row): # or col
                    if i == j:
                        self.data[i].append(1)
                    else:
                        self.data[i].append(0)
            # 高斯-约旦消元
            for i in range(self.row):
                t = self.data[i][i]
                for j in range(i , 2*self.row):
                    self.data[i][j] /= t
                for k in range(self.row):
                    if k == i:
                        continue
                    t = self.data[k][i]
                    for j in range(i , self.row*2):
                        self.data[k][j] -= t*self.data[i][j]
            # 复写,覆盖式
            for i in range(self.row):
                self.data[i] = deepcopy(self.data[i][self.row::])
            return self
        
        def frobenius_norm(self):
            res = Decimal(0)
            for row in self.data:
                for i in row:
                    res = res + i**2
            return res.sqrt()

        def __rmul__(self, other):
            if type(other) == float:
                other = Decimal(str(other))
            elif type(other) == str:
                other = Decimal(other)
            return super().__rmul__(other)

    class ParseType(Enum):
        coefficient = 1
        monomial = 2
    def __init__(self , polynomial: str , rawMode=False , raw={}):
        '''
        类中的所有与计算有关的数字类型均为Decimal类型
        self.func = {
            (1,3)   :   2,
            (1)     :   -1,
            ...
        }
        '''
        if rawMode: # 用于梯度和黑塞矩阵
            if not raw:
                raise ValueError("使用raw模式输入但是没有传入数据")
            self.dimension = raw["dimension"]
            self.func = raw["func"]
        else:
            self.dimension = 0 # 在decode_from_list中赋值，以最大的为准
            self.func = self.__decode_from_list(self.__string_to_polynomial(polynomial))
        self.gradient = None
        self.hessianMatrix = None

    def __str__(self):
        str = "f(X)="
        if self.func:
            for key , value in self.func.items():
                if value > 0:
                    str = str + '+'
                #if value != Decimal(1):
                str = str +  f"{value}"
                for index , j in enumerate(key):
                    if j != 0: # Decimal 0 和python的数字类型0可以比较，故不用Deicmal
                        # 即该单项式中存在当前变量
                        if j == Decimal("1"):
                            # 次数为1时不用写出
                            str = str + f"(x{index+1})"
                        else: # 即次数不为1
                            str = str + f"(x{index+1}^{j})"
            return str
        else:
            raise ValueError("类MathFunction没有被正确地初始化.")
    
    def __repr__(self):
        return str(self)

    def __mul__(self , other):
        func = deepcopy(self.func)
        dimension = self.dimension
        t = type(other)
        if t == Decimal or t == int or t == float:
            other = Decimal(str(other))
            for key in func.keys():
                func[key] = func[key]*other
            result = MathFunction("" , rawMode=True ,raw={"func":func , "dimension":dimension})
            return result
        elif isinstance(other , MathFunction): # 兼容子类
            # 将两者的func交叉相乘
            # dimension为最大者
            dimension = max(dimension , other.dimension)
            newFunc = {}
            for keyA , valueA in func.items():
                for keyB , valueB in other.func.items():
                    if len(keyA) > len(keyB):
                        newKey = list(keyA)
                        for index , i in enumerate(keyB):
                            newKey[index] += i
                    else:
                        newKey = list(keyB)
                        for index , i in enumerate(keyA):
                            newKey[index] += i
                    newValue = valueA*valueB
                    if newValue != 0:
                        newKey = tuple(newKey)
                        if newKey in tuple(newFunc.keys()):
                            newFunc[newKey] += newValue
                        else:
                            newFunc[newKey] = newValue
            if not newFunc.keys():
                newFunc[()] = 0
            result = MathFunction("" , rawMode=True , raw={"func":newFunc , "dimension":dimension})
            return result

    def __rmul__(self , other):
        return self.__mul__(other)
    
    def __add__(self , other):
        if not isinstance(other , MathFunction):
            raise ValueError(f"加法运算操作对象不是{type(self)}类型")
        newFunc = deepcopy(self.func)
        for key , value in other.func.items():
            if key in newFunc:
                newFunc[key] += value
            else:
                newFunc[key] = value
        # 再检查一遍，将0值删除
        delList = []
        for key , value in newFunc.items():
            if value == 0:
                delList.append(key)
        for i in delList:
            del newFunc[i]
        if not newFunc.keys():
            newFunc[()] = 0
        result = MathFunction("" , rawMode=True , raw={"func": newFunc , "dimension":max(self.dimension , other.dimension)})
        return result

    def __sub__(self , other):
        return self + other*(-1)

    @staticmethod
    def raw_column_matrix_x_to_list(rowMatrix: list):
        '''
        rowMatrix = [
            [x1],
            [x2],
            ... ...
        ]
        '''
        return [i[0] for i in rowMatrix]

    @staticmethod
    def parse_monomial(monomial: str):
        powerPattern = r"x\d+\^?-?\d*"
        coefficientPattern = r"[+-]?\d*\.?\d*"
        matches = re.findall(powerPattern, monomial)

        if matches:
            parseType = MathFunction.ParseType.monomial
            matches = re.split(r'[x\^]' , matches[0])
            matches = matches[1::]
            if len(matches) == 1: # 如果没有次数，既是1次
                matches.append("1")
        else:
            parseType = MathFunction.ParseType.coefficient
            matches = re.findall(coefficientPattern , monomial)
            matches = matches[0]
        return {"type": parseType , "data": matches}

    def __decode_from_list(self , li: str) -> dict:
        '''
        li: 传入的列表，格式为:  
        [
            ["-1" , "x1^2" , "x2"] # 表示 -x1^2*x2
            ...
        ]
        函数的存储形式：
        {(1,2):3} 表示： x1的一次方 乘以 x2的平方 系数为3
        '''
        res = {}
        for monomial in li:
            value = None
            rawKey = []
            for i in monomial:
                if not i:
                    continue
                parse = MathFunction.parse_monomial(i)
                if parse["type"] == MathFunction.ParseType.coefficient:
                    value = Decimal(parse["data"])
                else: #elif parse["type"] == MathFunction.ParseType.monomial:
                    rawKey.append(parse["data"])
            # 将键排序提取成tuple
            rawKey.sort(key=lambda x: x[0])
            key = [] # 最后要格式化成tuple
            while rawKey:
                j = rawKey.pop(0)
                while int(j[0]) - 1 != len(key): # x1的次数 在插入时， key列表应该是空的，即长度为 1-1 前者是x的下标。同理，在插入x5的次数时，key的长度应该是 5 - 1
                    key.append(Decimal(0))
                key.append(Decimal(j[1]))
            # 将key 列表转化为元组
            key = tuple(key)
            self.dimension = max(self.dimension , len(key))
            if value == None: # 没有解析到系数
                # 当系数为1时不用写系数
                value = Decimal(1)
            if key in res.keys():
                res[key] += value
            else:
                res[key] = value
        return res

    def __string_to_polynomial(self , s: str):
        '''
        return res = [
            ["-3" , "x^1"],
            ... ...
        ]
        '''
        res = []
        queue = deque()
        empty = False # 用于解析最后一个单项式及循环的退出
        filtered = False # 遇到左括号开启，右括号关闭（对于负次数）
        s = list(s)
        while True:
            if s:
                i = s.pop(0)
            else:
                empty = True
            if ( (empty) or i == "+" or i == "-") and (not filtered):
                # 出栈
                # 出栈结果是一个单项式
                monomial = [""]
                while queue:
                    # 系数、变量、次数
                    # 系数、变量及次数之间使用*号隔开
                    # -xn需要特殊处理
                    j = queue.popleft()
                    if j == "*":
                        monomial.append("")
                    else: # j != "*":
                        # 加到monomial最后一个
                        # 特殊处理-xn
                        if (j == "-" or j == '+') and queue[0] == "x":
                            monomial[-1] = j + "1"
                            # 为下一个，即xn开新的元素
                            monomial.append("")
                            pass
                        else:
                            monomial[-1] = monomial[-1] + j
                # 单项式解析完毕
                if monomial != [""]:
                    res.append(monomial)
            # 将正负号也进栈,左右括号不进栈
            if i == '(':
                filtered = True
            elif i == ')':
                filtered = False
            elif i != ' ': # 筛去空格
                queue.append(i)
            if empty:
                break
        return res

    # 功能函数
    def evaluate(self , x: list):
        '''
        x: list 按照index，传入各变量(0 对应 x1)的值,值为必须为float,程序在处理前会转化为Decimal
        '''
        if type(x) == self.DecimalMatrix: # 兼容直接传入列向量
            x = x.data
        if len(x) != self.dimension:
            raise ValueError(f"传入的向量与函数维度不符合：X的维度为{len(x)}而函数的维度为{self.dimension}")
        if type(x[0]) == list:  # 兼容列向量
            x = MathFunction.raw_column_matrix_x_to_list(x)
        x = [Decimal(str(i)) for i in x]
        res = Decimal("0")
        for powers , coefficient in self.func.items():
            monomialRes = Decimal("1")
            monomialRes = monomialRes * coefficient
            # 计算多项式中的单项式
            for index , power in enumerate(powers):
                if power != 0: # Decimal 0 和 python 0可以相比较，过滤0次相，减少开销
                    # Decimal 不能处理 0的n次方
                    if x[index] == 0: # Decimal 0 和 python 0可以相比较
                        # 如果要计算幂的变量的值为0，则该单项式为0
                        monomialRes = Decimal(0)
                        break
                    monomialRes = monomialRes * (x[index]**power)
            res = res + monomialRes
        return res

    def derivative(self , xIndex=0):
        rawData = {"dimension" : self.dimension , "func" : {}}
        # index : 0 -> x1 , 1 -> x2  ,... ...
        for powers , value in self.func.items():
            if xIndex > len(powers) - 1: # 原powers的存储格式为 只存到编号最大的变量，故可能存的长度少于维度
                continue # 跳过即可，因为求偏导为0
            newValue = powers[xIndex] * value
            if newValue == 0: # 求导结果为0
                continue # 跳过
            newPowers = list(powers)
            newPowers[xIndex] = newPowers[xIndex] - 1
            newPowers = tuple(newPowers)
            rawData["func"][newPowers] = newValue
        if rawData["func"] == {}: # 偏导为0
            rawData["func"] = {():Decimal(0)}
        newFunc = MathFunction("" , rawMode=True , raw=rawData)
        return newFunc

    def gradient_matrix(self):
        '''
        返回的式UniversalMatrix
        res.data =[
            [偏f/偏x1], # 类型为 MathFunction
            [偏f/偏x2],
            ... ...
        ]
        '''
        if self.gradient:
            return self.gradient
        res = []
        for index in range(self.dimension):
            newFunc = self.derivative(index)
            res.append(newFunc)

        matrix = [[i] for i in res]
        self.gradient = GenericMatrix(matrix)
        return self.gradient

    def evaluate_gradient(self , x: list):
        '''
        formate: 输出格式。str、decimal、float
        返回结果形式： 列向量
        MathFunction.DecimalMatrix [[1],[1] ...]
        '''
        # 不用检查维度是否对应，在evaluate中会检查
        if not self.gradient:
            self.gradient_matrix()
        res = []
        gradient = deepcopy(self.gradient)
        gradient.transpose()
        for i in gradient.data[0]: # 梯度的转置为行向量，行数为1
            res.append(i.evaluate(x))
        matrix = [[i] for i in res]
        return MathFunction.DecimalMatrix(matrix)

    def hessian_matrix(self):
        if self.hessianMatrix:
            return self.hessianMatrix
        if not self.gradient: # 若梯度函数没有求解，先求解梯度
            self.gradient_matrix()
        # 黑塞矩阵即对梯度再次求偏导，转置这些梯度的梯度，合并即可
        res = []
        gradient = deepcopy(self.gradient)
        gradient.transpose()
        for i in gradient.data[0]:
            gradientOfgradient = i.gradient_matrix()
            gradientOfgradient.transpose()
            res = res + gradientOfgradient.data
        self.hessianMatrix = GenericMatrix(res)
        return self.hessianMatrix

    def evaluate_hessian_matrix(self , x: list):
        '''
        x支持传入 列list ， 行list ， DecimalMatrix
        '''
        if not self.hessianMatrix:
            self.hessian_matrix()
        res = [[self.hessianMatrix.data[i][j].evaluate(x) for j in range(self.dimension)] for i in range(self.dimension)]
        return MathFunction.DecimalMatrix(res)

class ExtendedMathFunction(MathFunction):
    def __init__(self, polynomial, rawMode=False, raw={}):
        if not rawMode:
            # 拆解字符串
            # 以+ - 号拆解
            polynomial = filter_space(polynomial)
            func = self.__decode_to_list(polynomial)
            super().__init__("" , rawMode=True , raw={"func": func.func , "dimension": func.dimension})
        else:
            super().__init__(polynomial, rawMode, raw)

    @staticmethod
    def decode_folded_monomial(s: str):
        '''
        input: s= "-[x1+x2]^2" 或者 "-[x1+[x2+x3]]^2" 等
        output:  deque()
        '''
        originQue = deque(s)
        optQue = deque()
        if originQue[0] in "-+" and (not originQue[1].isnumeric()):
            # 第一个是-或者+号，且第二个不是数字
            char = originQue.popleft()
            if char == "-": # 负号要转义
                optQue.append("-1")
                optQue.append("*")
            # else: # 是正号，不用解析，丢弃即可
        # 将字符串解析为一系列操作,存储在操作队列中
        while originQue:
            item = originQue.popleft()
            if item == "[": # 多项式相乘的开始 (可能是多重折叠)
                sign = 1
                # 将多项式作为单独的变量
                s = ''
                while True:
                    item = originQue.popleft()
                    if item == "[":
                        sign += 1
                    elif item == ']':
                        sign -= 1
                        if sign == 0:
                            break
                    s += item
                optQue.append(s)
            elif item == "^":
                # 将^加入，后面应该是全数字，直接加入直到不是数字 （不允许负数）
                optQue.append(item)
                s = ""
                while originQue and originQue[0].isnumeric():
                    item = originQue.popleft()
                    s += item
                optQue.append(s)
            else: # 只剩数字
                s = ''
                s += item # 先把第一个加进去，可能是-+ 也可能是数字
                while originQue and originQue[0].isnumeric(): # 直到originque为空或者不是数字
                    s += originQue.popleft()
                optQue.append(s)
        return optQue

    @staticmethod
    def unfold(optQue) -> MathFunction:
        interQue = deque()
        while optQue:
            item = optQue.popleft()
            if item == "^":
                a = interQue.pop()
                b = optQue.popleft() # b是数字，但是已经转化为mathfunction类型了
                # 转换回纯数字 decimal
                b = b.func[()]
                b = int(b)
                for _ in range(b-1):
                    a = a*a
                interQue.append(a)
            else:
                interQue.append(item)
        # 运算*
        res = interQue.popleft()
        while interQue:
            item = interQue.popleft()
            if item == "*":
                b = interQue.popleft()
                res = res * b
            else:
                raise ValueError(f"理论上应该只会遍历到*号，但是此处是{item}")
        # 计算完毕
        return res

    def __decode_to_list(self , str):
        li = divide_by_plus_minus(str)
        # 现在li中存的是单个式子，如 x1 或 +x3 或者 -[x1+x2]^5
        # 需要将 -[x1+x2]^5 解析成 -1 * x1+x2 ^ 5 并运算
        for index , i in enumerate(li):
            # 将没有[]的式子直接解析为函数
            if "[" in i:
                # 将折叠的单项式展开为操作： [ "-1" , "*" , "[" , "x1" , "+" , "x2" , "]" "^" , "2" ]
                optQue = self.decode_folded_monomial(i)
                # 运算操作
                # 先将函数初始化
                for _index , value in enumerate(optQue):
                    if value == "*" or value == "^": # 操作符不解析
                        pass
                    else:
                        optQue[_index] = ExtendedMathFunction(value)
                # 运算
                res = self.unfold(optQue=optQue)
                li[index] = res
            else: # []不在i中
                # 直接初始化为函数类型
                li[index] = MathFunction(i)
        # 将多项式相加
        res = li[0]
        for i in range(1 , len(li)):
            res += li[i]
        return res

class FractionFunction(GenericFraction , MathFunction):
    def __init__(self, *args):
        if len(args) == 2:
            numerator = args[0]
            denominator = args[1]
        elif len(args) == 1:
            s = args[0]
            s = filter_space(s)
            # 处理正负号
            pre = ExtendedMathFunction("1")
            if s[0] in "+-" and  (not (s[1].isalpha() or s[1].isnumeric())):
                if s[0] == "-" :
                    pre = pre * (-1)
                s = s[1::]
            # 解析，使用//号表示，用{}表示范围，如{x1}//{x2} 表示x1/x2
            if "{" in s and (not self.is_monofraction(s)):
                # 可能是多个分式相加
                li = divide_by_plus_minus(s)
                res = FractionFunction("0")
                for i in li:
                    i = FractionFunction(i)
                    res = res + i
                numerator = res.numerator
                denominator = res.denominator
            else:
                # 单个式子
                if "//" in s: # 是分式
                    numerator , denominator = s.split("//")
                    numerator = re.split(r"[{}]" , numerator)[1]
                    denominator = re.split(r"[{}]" , denominator)[1]
                    pass
                else: # 不是分式，初始化分母为1
                    numerator = s
                    denominator = "1"
                numerator = ExtendedMathFunction(numerator)
                denominator = ExtendedMathFunction(denominator)
            numerator = numerator * pre
        else:
            raise ValueError("初始化分式函数的输入参数数量错误")
        super().__init__(numerator, denominator)
        self.gradient = None
        self.hessianMatrix = None
        self.dimension = max(self.numerator.dimension , self.denominator.dimension)
        self.numerator.dimension = self.dimension
        self.denominator.dimension = self.dimension

    @staticmethod
    def is_monofraction(s: str):
        filterSign = 0
        for i in s:
            if i == "{":
                filterSign += 1
            elif i == "}":
                filterSign -= 1
            elif filterSign:
                continue
            elif i == "+" or i == "-":
                return False
        else:
            return True

    def __str__(self):
        result = "f(X)={"
        result += f"{self.numerator}"
        result += "}//{"
        result += f"{self.denominator}"
        result += "}"
        return result

    def __repr__(self):
        return str(self)

    # 功能函数
    def evaluate(self , x: list):
        n = self.numerator.evaluate(x)
        d = self.denominator.evaluate(x)
        try:
            result = n/d
        except ZeroDivisionError:
            result = Decimal(math.inf)
        return result

    def derivative(self , xIndex=0):
        # 分式求导法则，分母平方，分子为：上导*下不导 - 下导*上不导
        denominator = self.denominator
        numerator = self.numerator
        derivativedDenominator = denominator.derivative(xIndex)
        derivativedNumerator = numerator.derivative(xIndex)
        numerator = derivativedNumerator*denominator - numerator*derivativedDenominator
        denominator = denominator*denominator
        return type(self)(numerator , denominator)

    def update_dimension(self , newDimension):
        self.numerator.dimension = newDimension
        self.denominator.dimension = newDimension

    def __truediv__(self, other):
        if isinstance(other , FractionFunction):
            return super().__truediv__(other)
        elif isinstance(other , MathFunction):
            numerator = self.numerator
            denominator = self.denominator * other
            return type(self)(numerator , denominator)
        elif isinstance(other , int):
            other = Decimal(other)
        elif isinstance(other , float):
            other = Decimal(str(other))

        if isinstance(other , Decimal):
            denominator = self.denominator * other
            numerator = self.numerator
            return type(self)(numerator , denominator)

    def __rtruediv__(self , other):
        if isinstance(other , FractionFunction):
            denominator = self.numerator * other.denominator
            numerator = self.denominator * other.numerator
        elif isinstance(other , MathFunction):
            denominator = self.numerator
            numerator = self.denominator * other
        elif isinstance(other , int):
            other = Decimal(other)
        elif isinstance(other , float):
            other = Decimal(str(other))

        if isinstance(other , Decimal):
            numerator = self.denominator * other
            denominator = self.numerator
        return type(self)(numerator , denominator)
    
    def simplify(self):
        temp = [inf for _ in range(self.dimension)]
        for i in self.numerator.func.keys():
            for index in range(self.dimension):
                if index < len(i):
                    value = i[index]
                    temp[index] = min(temp[index], value)
                else:
                    temp[index] = min(temp[index], Decimal(0))
        for i in self.denominator.func.keys():
            for index in range(self.dimension):
                if index < len(i):
                    value = i[index]
                    temp[index] = min(temp[index], value)
                else:
                    temp[index] = min(temp[index], Decimal(0))
        g = 0
        for i in self.numerator.func.values():
            g = gcd(g , abs(i))
        for i in self.denominator.func.values():
            g = gcd(g , abs(i))
        newNumerator = {}
        newDenominator = {}
        for key, value in self.numerator.func.items():
            newKey = list(key)
            for index , j in enumerate(newKey):
                newKey[index] = j - temp[index]
            newKey = tuple(newKey)
            newValue = value // g
            newNumerator[newKey] = newValue
        for key, value in self.denominator.func.items():
            newKey = list(key)
            for index, j in enumerate(newKey):
                newKey[index] = j - temp[index]
            newKey = tuple(newKey)
            newValue = value // g
            newDenominator[newKey] = newValue
        self.numerator.func = newNumerator
        self.denominator.func = newDenominator