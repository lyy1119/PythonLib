from decimal import Decimal

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


class MathFunction:
    class UniversalMatrix:
        def __init__(self , li: list , check=True):
            '''
            矩阵格式：
            [
                [],
                [],
                ... ...
            ]
            无论有几行，都要写成上述格式
            直接使用li，若数据量太大，使用参数check=False关闭矩阵格式检查（每行长度是否相同）
            '''
            # check
            if check:
                # 开始检查
                # 1.检查li的元素是否为list
                if type(li[0]) == list:
                    pass
                else:
                    raise ValueError("用以初始化通用矩阵类的输入列表格式错误！")
                col = len(li[0])
                for row in li:
                    if len(row) != col:
                        raise ValueError("用以初始化通用矩阵类的输入列表的各行长度不相同！")
            from copy import deepcopy
            # 使用deepcopy，防止可能的嵌套结构
            self.data = deepcopy(li)
            self.row = len(self.data)
            self.col = len(self.data[0])

        def row_transformation(self , rowA: int , rowB: int):
            '''
            不可逆操作
            会修改原矩阵
            '''
            if rowA >= self.row or rowB > self.row:
                raise ValueError("所要交换的行超出矩阵范围")
            from copy import deepcopy
            rowAData = deepcopy(self.data[rowA])
            rowBData = deepcopy(self.data[rowB])
            self.data[rowB] = rowAData
            self.data[rowA] = rowBData

        def transpose(self):
            self.data = [[self.data[i][j] for i in range(self.row)] for j in range(self.col)]
            return self


        def __str__(self):
            res = ''
            for row in self.data:
                for i in row:
                    res = res + f"{i}" + "\t"
                res = res + "\n"
            return res

        def __mul__(self , other):
            """
            return res # type=MathFunction.UniversialMatrix
            """
            if type(other) != type(self):
                raise ValueError("乘号右侧必须是通用矩阵类或其子类！")
            front = self
            after = other

            if front.col != after.row:
                # 矩阵乘法检测
                raise ValueError("两矩阵相乘必须满足：左矩阵列数 = 右矩阵行数")
            res = [[None for i in range(after.col)] for i in range(front.row)]
            for row in range(front.row):
                for colum in range(after.col):
                    temp = front.data[row][0] * after.data[0][colum]
                    for i in range(1 ,front.col): # or after.row
                        # 从下标1开始，便于确定类型
                        temp = temp + front.data[row][i] * after.data[i][colum]
                    res[row][colum] = temp
            return MathFunction.UniversalMatrix(res)

        def __rmul__(self , other):
            # 左乘
            newData = [[j*other for j in i] for i in self.data]
            return type(self)(newData)

        def __add__(self , other):
            if self.row == other.row and self.col == other.col:
                newData = [[self.data[i][j] + other.data[i][j] for j in range(self.col)] for i in range(self.row)]
                return type(self)(newData)
            else:
                raise ValueError(f"相加两矩阵的行列数目不对应，前者row={self.row},col={self.col} , 后者row={other.row},col={self.col}")

    class DecimalMatrix(UniversalMatrix):
        def __init__(self, li, check=True):
            if type(li[0][0]) != Decimal:
                li = [[Decimal(j) for j in i] for i in li]
            super().__init__(li, check)
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
                from copy import deepcopy
                self.data[i] = deepcopy(self.data[i][self.row::])
            return self

    from enum import Enum
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
            self.dimansion = raw["dimansion"]
            self.func = raw["func"]
        else:
            self.dimansion = 0 # 在decode_from_list中赋值，以最大的为准
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
        import re
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
            self.dimansion = max(self.dimansion , len(key))
            if value == None: # 没有解析到系数
                # 当系数为1时不用写系数
                value = Decimal(1)
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
        queue = []
        empty = False # 用于解析最后一个单项式及循环的退出
        filtered = False # 遇到左括号开启，右括号关闭（对于负次数）
        s = list(s)
        while True:
            if s:
                i = s.pop(0)
            else:
                empty = True
            if (i == "+" or i == "-" or (empty)) and (not filtered):
                # 出栈
                # 出栈结果是一个单项式
                monomial = [""]
                while queue:
                    # 系数、变量、次数
                    # 系数、变量及次数之间使用*号隔开
                    # -xn需要特殊处理
                    j = queue.pop(0)
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
        x: list 按照index，传入各变量(0 对应 x1)的值,值为必须为float,程序在处理前会转化为Deciaml
        '''
        if len(x) != self.dimansion:
            raise ValueError(f"传入的向量与函数维度不符合：X的维度为{len(x)}而函数的维度为{self.dimansion}")
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
        return {"str" : str(res) , "float" : float(res) , "Decimal" : res }

    def gradient_matrix(self):
        '''
        返回的式UniversalMatrix
        res.data =[
            [偏f/偏x1], # 类型为 MathFunction
            [偏f/偏x2],
            ... ...
        ]
        '''
        if not self.func:
            raise ValueError("函数未正确初始化.")
        if self.gradient:
            return self.gradient
        res = []
        for index in range(self.dimansion):
            rawData = {"dimansion" : self.dimansion , "func" : {}}
            # index : 0 -> x1 , 1 -> x2  ,... ...
            for powers , value in self.func.items():
                if index > len(powers) - 1: # 原powers的存储格式为 只存到编号最大的变量，故可能存的长度少于维度
                    continue # 跳过即可，因为求偏导为0
                newValue = powers[index] * value
                if newValue == 0: # 求导结果为0
                    continue # 跳过
                newPowers = list(powers)
                newPowers[index] = newPowers[index] - 1
                newPowers = tuple(newPowers)
                rawData["func"][newPowers] = newValue
            if rawData["func"] == {}: # 偏导为0
                rawData["func"] = {():Decimal(0)}
            newFunc = MathFunction("" , rawMode=True , raw=rawData)
            res.append(newFunc)

        matrix = [[i] for i in res]
        self.gradient = MathFunction.UniversalMatrix(matrix)
        return self.gradient

    def evaluate_gradient(self , x: list , format="float"):
        '''
        formate: 输出格式。str、decimal、float
        返回结果有两种形式：
        "raw" : [1 , 1 , 1 , ...]
        "matrix": MathFunction.DeciamlMatrix [[1],[1] ...]
        '''
        # 不用检查维度是否对应，在evaluate中会检查
        if not self.gradient:
            self.gradient_matrix()
        res = []
        from copy import deepcopy
        gradient = deepcopy(self.gradient)
        gradient.transpose()
        for i in gradient.data[0]: # 梯度的转置为行向量，行数为1
            res.append(i.evaluate(x)[format])
        matrix = [[i] for i in res]
        return {"raw" : res , "matrix" : MathFunction.DecimalMatrix(matrix)}

    def hessian_matrix(self):
        if self.hessianMatrix:
            return self.hessianMatrix
        if not self.gradient: # 若梯度函数没有求解，先求解梯度
            self.gradient_matrix()
        # 黑塞矩阵即对梯度再次求偏导，转置这些梯度的梯度，合并即可
        res = []
        from copy import deepcopy
        gradient = deepcopy(self.gradient)
        gradient.transpose()
        for i in gradient.data[0]:
            gradientOfgradient = i.gradient_matrix()
            gradientOfgradient.transpose()
            res = res + gradientOfgradient.data
        self.hessianMatrix = MathFunction.UniversalMatrix(res)
        return self.hessianMatrix

    def evaluate_hessian_matrix(self , x: list):
        res = [[self.hessianMatrix.data[i][j].evaluate(x)["Decimal"] for j in range(self.dimansion)] for i in range(self.dimansion)]
        return MathFunction.DecimalMatrix(res)

if __name__ == "__main__":
    print("这是MathFunction类库的测试单元！")
    print("测试开始")
    print("===================")
    print("（1）测试从字符串解析函数")
    s = "x1^2 - 8 + 5*x2^7*x1^2 + x1^(-5) + x5 + x1*x2*x3"
    print(f"s = {s}")
    print("function=MathFunction(s)")
    function = MathFunction(s)
    print("print(function)")
    print("[Out]: " , end='')
    print(function)
    print(function.func)
    print(f"函数维度（变量的数量）:{function.dimansion}")
    print("测试成功！")

    print()

    print("（2）测试求函数值")
    print("value = function.evaluate([1,1,1,2,1])")
    x = [
        [1],
        [1],
        [1],
        [2],
        [1]
    ]
    value = function.evaluate(x)
    print(r'print(value["str"])')
    print("[Out]: " , end='')
    print(value["str"])
    print("测试成功!")

    print()

    print("（3）测试求梯度函数")
    gradient = function.gradient_matrix()
    print("gradf(X)=")
    print(gradient)
    print("x = [1,1,1,1,1]")
    x = [1,1,1,1,1]
    r = function.evaluate_gradient(x)["matrix"]
    print("梯度为X=")
    print(r)
    print("测试成功!")

    print()

    print("（4）测试高斯-约旦消元法求逆矩阵")
    print("测试用例1：")
    print("矩阵：")
    a = [
        [Decimal(1) , Decimal(2) , Decimal(3)],
        [Decimal(2) , Decimal(2) , Decimal(1)],
        [Decimal(3) , Decimal(4) , Decimal(3)]
    ]
    b = MathFunction.DecimalMatrix(a)
    from copy import deepcopy
    c = deepcopy(b)
    print(b)
    print("创建MathFunction.DecimalMatrix类变量")
    print(r'MathFunction.DecimalMatrix(a)')
    print("逆矩阵为：")
    b.inverse()
    print(b)
    print(c)

    print("测试用例2：")
    print("矩阵：")
    a = [
        [Decimal(0) , Decimal(0) , Decimal(1)],
        [Decimal(0) , Decimal(1) , Decimal(0)],
        [Decimal(1) , Decimal(0) , Decimal(0)]
    ]
    b = MathFunction.DecimalMatrix(a)
    print(b)
    print("创建MathFunction.DecimalMatrix类变量")
    print(r'MathFunction.DecimalMatrix(a)')
    print("逆矩阵为：")
    b.inverse()
    print(b)

    print("（5）测试矩阵转置:")
    a = [[1,2,3,4,5]]
    b = MathFunction.DecimalMatrix(a)
    print("原矩阵为：")
    print(b)
    print("转置后的矩阵为：")
    b.transpose()
    print(b)
    print(f"转置后的矩阵变量类型为：{type(b)}")

    print()

    print("（6）测试计算黑塞矩阵函数：")
    print(function.hessian_matrix())
    print("测试成功！")

    print()

    print("（7）测试黑塞矩阵数值计算：")
    print("x = [1,1,1,1,1]")
    x = [1,1,1,1,1]
    print("黑塞矩阵的值为:")
    print(function.evaluate_hessian_matrix(x))
    print("测试成功！")

    print("（8）测试矩阵相乘:")
    print("矩阵1：")
    a = MathFunction.DecimalMatrix([[1,2,3]])
    print(a)
    b = MathFunction.DecimalMatrix([[1],[8],[3]])
    print("矩阵2：")
    print(b)
    print("a*b结果：")
    print(a*b)
    print("测试成功！")

    print()

    print("（9）测试矩阵数乘:")
    print("原矩阵：")
    a = MathFunction.DecimalMatrix([[1,2,3]])
    print(a)
    print("5*a:")
    print(5*a)
    print("测试成功！")

    print()

    print("（10）测试矩阵相加:")
    a = MathFunction.DecimalMatrix([[1,2,3]])
    b = MathFunction.DecimalMatrix([[4,5,6]])
    print("矩阵1：")
    print(a)
    print("矩阵2：")
    print(b)
    print("两矩阵相加 a+b:")
    print(a+b)
    print("测试成功")