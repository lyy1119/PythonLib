from mathFunction import MathFunction

def determine_search_interval(function: MathFunction , x0: list , t0: float , s: list):
    '''
    function: 函数
    x0: 起点
    t0: 初始步长
    s: 搜索方向
    return {"float" : [a , b] , "Decimal" : [a , b]}
    '''
    