from scipy.optimize import minimize
import numpy as np
from sympy import *
from sympy.core.numbers import NaN
from copy import deepcopy


def logBarMethod(func: str, restrictions: list, start_point: tuple = tuple(), accuracy:float = 10**(-6), max_steps: int=500):
    '''
    Решение задачи оптимизации для функции методом логарифмических барьеров.
    
    Parameters
    ----------
    func : str
        Функция оптимизации.
    equality : list
        Список заданных линейных ограничений.
    x0 : tuple
        Начальная точка.
    tol : int, default=5
        Кол-во цифр после запятой после округления.
    Returns 
    ----------
    np.array(next_point), res_new:
        Найденная точка и значение функции в ней.
    '''
    tao = 1
    v = 10
    for i in range(len(restrictions)):
        if '>' in restrictions[i]:
            restrictions[i] = restrictions[i][:restrictions[i].index('>')].replace(' ', '')
        else:
            restrictions[i] = restrictions[i][:restrictions[i].index('<')].replace(' ', '')

    phi = f'{tao}*({func})'
    for exp in restrictions:
        phi += f' - log({exp})'
        
    X = Matrix([sympify(phi)])
    symbs = list(sympify(phi).free_symbols)

    if len(start_point) == 0:
        start_point = first_phase(restrictions, symbs)
    if start_point == 'Введенные ограничения не могут использоваться вместе':
        return start_point
    elif start_point == 'Невозможно подобрать внутреннюю точку для данного метода':
        return start_point

    try:
        res = sympify(func).subs(list(zip(symbs, start_point)))
    except:
        return 'Введена первоначальная точка, которая не удовлетворяет неравенствам'
    Y = Matrix(list(symbs))
    
    df = X.jacobian(Y).T
    ddf = df.jacobian(Y)
    
    lst_for_subs =  list(zip(symbs, start_point))
    dfx0 = df.subs(lst_for_subs)
    ddfx0 = ddf.subs(lst_for_subs)
    
    xk = ddfx0.inv() @ dfx0
    next_point = [start_point[i]-xk[i] for i in range(len(start_point))]
    tao = tao*v
    
    
    res_new = sympify(func).subs(list(zip(symbs, next_point)))
    if type(res_new) == NaN:
        return np.array(start_point), res
        
    steps = 1
    while abs(res_new - res) > accuracy and max_steps > steps:
        phi = f'{tao}*({func})'
        for exp in restrictions:
            phi += f' - log({exp})'
        
        X = Matrix([sympify(phi)])
        symbs = list(sympify(phi).free_symbols)
        Y = Matrix(list(symbs))

        df = X.jacobian(Y).T
        ddf = df.jacobian(Y)

        lst_for_subs =  list(zip(symbs, start_point))
        dfx0 = df.subs(lst_for_subs)
        ddfx0 = ddf.subs(lst_for_subs)

        xk = ddfx0.inv() @ dfx0
        old_point = deepcopy(next_point)
        next_point = [next_point[i]-xk[i] for i in range(len(next_point))]
        res = deepcopy(res_new)
        res_new = sympify(func).subs(list(zip(symbs, next_point)))
        if type(res_new) == NaN:
            return np.array(old_point), res

        tao = tao*v
        steps += 1
        

    return np.array(next_point), res_new


def first_phase(restrictions: list, symbs: list) -> tuple:
    
    s = 1000
    restrictions_sympy = [sympify(i) for i in restrictions]
    res_functions = []
    for i in range(100, -100, -1):
        if s >= 0:
            x = [i for j in range(len(symbs))]
            s = max([expr.subs(list(zip(symbs, x))) for expr in restrictions_sympy])
        
    if s < 0:
        return x
    elif s > 0:
        return 'Введенные ограничения не могут использоваться вместе'
    elif s == 0:
        return 'Невозможно подобрать внутреннюю точку для данного метода'
    
    
def Newton(func: str, equality: list, x0: tuple, tol=5):
    """
    Решение задачи оптимизации  для функции с
    ограничениями типа равенства методом Ньютона.
    
    Parameters
    ----------
    func : str
        Функция оптимизации.
    equality : list
        Список заданных линейных ограничений.
    x0 : tuple
        Начальная точка.
    tol : int, default=5
        Кол-во цифр после запятой после округления.
    Returns 
    ----------
    res['x'], res['fun']:
        Найденная точка и значение функции в ней.
    """
    try:
        func = sympify(func)
        equality = [sympify(eq.partition('=')[0]) for eq in equality]
    except SympifyError:
        print('Неверно заданы функции')
    func_c = lambda x: func.subs(dict(zip(func.free_symbols, x)))
    eq_func = lambda x: [us.subs(dict(zip(us.free_symbols, x))) for us in equality]
    eq_constraints = {'type': 'eq',
                      'fun': eq_func}
    res = minimize(func_c, x0, method='SLSQP', constraints=eq_constraints)
    return res['x'].round(tol), round(res['fun'], tol)

def get_pain(first_line, second_line, third_line, A):
    """
    Вспомогательная функция для записи матрицы.
    Parameters
    ----------
    first_line : list
        Первая строка матрицы.
    second_line : list
        Вторая строка.
    third_line : list
        Третья строка.
    A : list
       Список коэффициентов для переменных в ограничении
       типа равенства.
    Returns
    ----------
    pain: list
        Матрица.
    """
    pain = []
    for line in [first_line, second_line, third_line]:
        if line == third_line and not A:
            continue
        pain_ = []
        pain_.append([l[i] for l in line for i in range(l.shape[1])]) 
        pain_.append([l[i] for l in line for i in range(l.shape[1], len(l))])
        pain.extend(pain_)
    return pain


def get_first_line(F0, Fi, Y, A, n, us_n):
    """
    Вычисляет первую строку для матрицы.
 
    Returns
    ----------
    first_line: list
        Первая строка матрицы.
    """
    
    first_line = [F0.jacobian(Y).T.jacobian(Y)]
    s = Matrix(np.zeros(shape=(n, n)))
    for u_n in us_n:
        s += Matrix([u_n]).jacobian(Y).jacobian(Y)
    first_line[0] += s
    first_line.append(Fi.jacobian(Y).T)
    if A:
        first_line.append(A.T)
    return first_line


def get_second_line(diag_lam, Fi, Y, A, m, n, p):
    """
    Вычисляет вторую строку для матрицы.
 
    Returns
    ----------
    second_line: list
        Вторая строка матрицы.
    """
    
    second_line = [diag_lam@Fi.jacobian(Y),
                   diag(*Fi, unpack=True)]
    if A:
        second_line.append(Matrix(np.zeros((m, (n+m+p-2*n)))))
    return second_line


def get_right_pain(F0, Y, Fi, lmbd, A, mu, diag_lam, e, Axb):
    """
    Вычисляет правую часть системы линейных
    уравнений из ужасного условия.
    """
    pain1 = [F0.jacobian(Y).T + Fi.jacobian(Y).T@lmbd]
    if A:
        pain1[0] = pain1[0] + A.T@mu
    pain2 = [diag_lam@Fi + e]
    pain3 = [Axb]
    pain = []
    pain.extend(pain1)
    pain.extend(pain2)
    if A:
        pain.extend(pain3)
    right_pain = Matrix(pain)
    return right_pain


def search_point(x0, lmbd, mu, var,
                 left_pain, right_pain, 
                 n, m, p, Fi, func, tol, A):
    """
    Выполняет поиск точки для заданных условий.
    Returns
    ----------
    point: np.array
        Найденная точка.
    """
    diff_f = 1000
    point_old = x0
    point = x0

    Lmbd = np.array([1 for i in lmbd.free_symbols])
    MU = np.array([1 for i in mu.free_symbols])

    fear_left = dict(zip(var + list(lmbd.free_symbols),
                    list(point) + list(Lmbd)))
    fear_right = dict(zip(var + list(lmbd.free_symbols) + list(mu.free_symbols),
                    list(point) + list(Lmbd) + list(MU)))

    while diff_f >= tol:
        left_pain_inv = left_pain.subs(fear_right).inv()
        right_pain_new = -1*right_pain.subs(fear_right)
        horror = left_pain_inv @ right_pain_new
        horror = horror.subs(fear_right)

        if A:
            dx, dl, dm  = [horror[i: i+n] for i in range(0, horror.shape[0], n)]
        else:
            dx, dl = [horror[i: i+n] for i in range(0, horror.shape[0], n)]
        alpha_p = 1
        point_old = point
        point = np.array(point_old) + alpha_p * np.array(dx)
        fear_right = dict(zip(var + list(lmbd.free_symbols) + list(mu.free_symbols),
                              list(point) + list(Lmbd) + list(MU)))
        panic = sum(1 for el in Fi.subs(fear_right) if el <= 0)
        while panic < Fi.shape[0]:
            alpha_p *= 0.5
            point = np.array(point_old) + alpha_p * np.array(dx)
            point = point.subs(fear_right)
            fear_right = dict(zip(var + list(lmbd.free_symbols) + list(mu.free_symbols),
                              list(point) + list(Lmbd) + list(MU)))
            panic = sum(1 for el in Fi.subs(fear_right) if el <= 0)
        thing = []
        for i in range(len(dl)):
            if dl[i] < 0:
                thing.append(-0.9 * Lmbd[i] / dl[i])
        if thing:
            alpha_d = min(1, *thing)
        else:
            alpha_d = 1
        Lmbd = np.array([Lmbd[i] + alpha_d*dl[i] for i in range(len(Lmbd))])
        MU = 0.1 * MU  
        fear_right_old = dict(zip(var + list(lmbd.free_symbols) + list(mu.free_symbols),
                              list(point_old) + list(Lmbd) + list(MU)))
        fear_right = dict(zip(var + list(lmbd.free_symbols) + list(mu.free_symbols),
                              list(point) + list(Lmbd) + list(MU)))
        diff_f = abs(func.subs(fear_right) - func.subs(fear_right_old))
        return point, func.subs(fear_right)


def inequality(func, x0, us, a, b, tol=10**-5):
    """
    Решение задачи оптимизации для функции с
    ограничениями типа неравенства 
    методом прямой двойной внутренней точки.
    Parameters
    ----------
    func : str
        Функция для оптимизации.
    x0 : list
        Начальная точка.
    us : list
        Список заданных линейных ограничений.
        Варианты ввода: '4*x - 4 = 0' или '4*x - 4 < 0'
    a : list
        Список коэффициентов для переменных в ограничении
        типа равенства.
    b : list
        Список свободных членов для ограничения типа равенства.
    tol : float, default=10**-5
        Критерий остановы.
    Returns 
    ----------
    point, y:
        Найденная точка и значение функции в ней.     
    """
    try:
        func = sympify(func)
        us_eq = [sympify(u.partition('=')[0]) for u in us if '<' not in u]
        us_n = [sympify(u.partition('<')[0]) for u in us if '<' in u]
        var = list(func.free_symbols)
        F0 = Matrix([func])
        Fi = Matrix(*[us_n])
        Y = Matrix(var)
        n = len(var)  # колич переменных
        m = len(us_n)  # неравенство 
        p = len(us_eq)  # равенство
        lmbd = Matrix([f'lambda{i}' for i in range(m)])
        diag_lam = diag(*lmbd, unpack=True)
        mu = Matrix([f'mu{i}' for i in range(p)])
        v = 0.1
        A = Matrix(a)
        B = Matrix(b)
        Axb = Matrix(np.zeros(shape=B.shape))
        e = Matrix(np.ones(shape=len(us_n)))
        diag_f = diag(*Fi, unpack=True)

        first_line = get_first_line(F0, Fi, Y, A, n, us_n)
        second_line = get_second_line(diag_lam, Fi, Y, A, m, n, p)
        third_line = [A, Matrix(np.zeros((p, m))), Matrix(np.zeros((p, p)))]

        left_pain = Matrix(get_pain(first_line, second_line, third_line, A))
        right_pain = get_right_pain(F0, Y, Fi, lmbd, A, mu, diag_lam, e, Axb)

        point, y = search_point(x0, lmbd, mu, var,
                                left_pain, right_pain, 
                                n, m, p, Fi, func, tol, A)
        return point, y
    except:
        str1 = 'К сожалению, решить не вышло, попробуйте изменить входные данные'
        print(str1)
        raise


