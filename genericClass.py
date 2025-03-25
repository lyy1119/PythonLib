class Matrix:
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
            self.col , self.row = self.row , self.col
            return self

        def __str__(self):
            # 对齐
            # 计算每列最大宽度
            col_widths = [max(len(str(item)) for item in col) for col in zip(*self.data)]
            res = '\n'.join(['| ' + ' '.join(f"{str(item):<{col_widths[i]}}" for i, item in enumerate(row)) + ' |' for row in self.data])
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
            return type(self)(res)

        def __rmul__(self , other):
            # 左乘
            newData = [[j*other for j in i] for i in self.data]
            return type(self)(newData)

        def __truediv__(self , other):
            from copy import deepcopy
            newData = deepcopy(self.data)
            for i in range(self.row):
                for j in range(self.col):
                    newData[i][j] /= other
            return type(self)(newData)

        def __neg__(self):
            return -1*self

        def __add__(self , other):
            if self.row == other.row and self.col == other.col:
                newData = [[self.data[i][j] + other.data[i][j] for j in range(self.col)] for i in range(self.row)]
                return type(self)(newData)
            else:
                raise ValueError(f"相加两矩阵的行列数目不对应，前者row={self.row},col={self.col} , 后者row={other.row},col={self.col}")
        def __sub__(self , other):
            return self + (-other)

class Fraction:
    def __init__(self , numerator , denominator):
        self.numerator = numerator
        self.denominator = denominator
    
    def __str__(self):
        return f"{self.numerator}/{self.denominator}"

    def __add__(self , other):
        if type(other) != type(self):
            raise ValueError(f"运算类型不相同，{type(self)}+{type(other)}")
        numerator = self.numerator * other.denominator + self.denominator * other.numerator
        denominator = self.denominator * other.denominator
        return type(self)(numerator , denominator)

    def __mul__(self , other):
        numerator = self.numerator
        denominator = self.denominator
        if type(other) != type(self):
            # 可能是类似数乘操作
            numerator = numerator * other
        else:
            # 分数相乘
            numerator = numerator * other.numerator
            denominator = denominator * other.denominator
        return type(self)(numerator , denominator)

    def __rmul__(self , other):
        return self.__mul__(other)

    def __neg__(self):
        return -1*self

def transpose(matrix: Matrix):
    from copy import deepcopy
    res = deepcopy(matrix)
    res.transpose()
    return res