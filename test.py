# 库的测试程序

from lyy19Lib import *
from lyy19Lib.mathFunction import *
from lyy19Lib.convexOptimization import *

def test_mathfunction():
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
    print(f"函数维度（变量的数量）:{function.dimension}")
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
    x = MathFunction.DecimalMatrix(x)
    value = function.evaluate(x)
    print(r'print(value["str"])')
    print("[Out]: " , end='')
    print(value)
    print("测试成功!")

    print()

    print("（3）测试求梯度函数")
    gradient = function.gradient_matrix()
    print("gradf(X)=")
    print(gradient)
    print("x = [1,1,1,1,1]")
    x = [1,1,1,1,1]
    r = function.evaluate_gradient(x)
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
    print("a/5:")
    print(a/5)
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

    print()

    print("（11）矩阵负号运算测试：")
    a = MathFunction.DecimalMatrix([[1,2,3]])
    print("原矩阵a:")
    print(a)
    print("-a:")
    print(-a)

    print()

    print("（12）测试mathfunction的乘法运算")
    fa = MathFunction("x1^2 + 2*x2")
    print(f"fa = {fa}")
    fb = MathFunction("2 + x3^5 - x1 + 4*x4^(-1)")
    print(f"fb = {fb}")
    print("f = fa*fb")
    print(fa*fb)

    print()

    fa = MathFunction("x1+x2")
    print(f"fa = {fa}")
    print("f = fa*fa")
    print(fa*fa)

    print()

    print(f"5*fa")
    print(5*fa)
    print(f"函数维度:{(5*fa).dimension}")

    print()

    print("（13）测试函数加法")
    fa = MathFunction("x1+x2^2+3")
    fb = MathFunction("4-x1+x2^2+x6^2")
    print(f"fa = {fa}")
    print(f"fb = {fb}")
    print(f"fa + fb = {fa+fb}")
    print(f"维度:{(fa+fb).dimension}")

    print("（14）测试不展开的函数式")
    s = "[x1+x2]*[x3+x4]"
    print(f"s={s}")
    f = ExtendedMathFunction(s)
    print(f"f= {f}")
    s = "[x1+1]^2"
    print(f"s={s}")
    f = ExtendedMathFunction(s)
    print(f"f= {f}")

def test_convexOptimization():
    print("Decimal精度：" , end="")
    print(getcontext().prec)  # 默认输出 28
    print()
    print("这是无约束优化的测试程序.")
    print()
    # 寻找区间测试
    print("寻找搜索区间测试")
    print()
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
    q.solve(MethodType.goldenSection)
    print(q.res)
# 二次插值测试 
    print("二次插值测试")
    q.searchInterval = [Decimal(-3) , Decimal(5)]
    q.solve(MethodType.quadraticInterpolation)
    print(q.res)

# 坐标轮换测试
    print("坐标轮换法测试.")
    q = MultidimensionOptimization(
        "4 + 4.5*x1 - 4*x2 + x1^2 + 2*x2^2 - 2*x1*x2 + x1^4 - 2*x1^2*x2",
        [0 , 0],
        0.1,
        0.1
    )
    q.solve()
    print(q.res)


    q = MultidimensionOptimization(
        "x1^2 + 2*x2^2",
        [0 , 0],
        0.01,
        0.01
    )
    q.solve(method=MethodType.gradientDescent)
    print(q.res)

# 阻尼牛顿法
    print("阻尼牛顿法测试.")
    q = MultidimensionOptimization(
        "x1^2 - x1*x2 + x2^2 + 2*x1 - 4*x2",
        [2 , 2],
        0.01,
        0.01
    )
    q.solve(method=MethodType.dampedNewton)
    print(q.res)

# 共轭方向法
    print("共轭方向法测试")
    q = MultidimensionOptimization(
        "2*x1^2 + 2*x1*x2 + 2*x2^2",
        [10 , 10],
        0.01,
        0.01
    )
    q.oneDimensionProblemMethod = MethodType.quadraticInterpolation
    q.solve(method=MethodType.conjugateDirection)
    print(q.res)
    print(q.read_logs())

# powell
    print("powell法测试")
    q = MultidimensionOptimization(
        "11*x1^2 + 11*x2^2 + 18*x1*x2 - 100*x1 - 100*x2 + 250",
        [0 , 0],
        0.01,
        0.01
    )
    q.solve(method=MethodType.powell)
    print(q.res)

# dfp
    print("dfp法测试")
    q = MultidimensionOptimization(
        "4*x1^2 + x2^2 - 40*x1 - 12*x2 + 136",
        [8 , 9],
        0.001,
        0.01
    )
    # q.oneDimensionProblemMethod=MethodType.quadraticInterpolation
    q.solve(method=MethodType.dfp)
    print(q.res)
    print("bfgs法测试")
    q.solve(method=MethodType.bfgs)
    print(q.res)



print("""
===========================
这是库lyy19Lib的测试程序
0. 测试所有库
1. 测试mathFunction
2. 测试convexOptimization
""")

n = int(input("input:"))

test = [
    test_mathfunction,
    test_convexOptimization
]

if n == 0:
    for i in test:
        i()
else:
    test[n-1]()