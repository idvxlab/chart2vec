import functools

def memoize(func):
    cache = {}  # 用于缓存结果的字典

    @functools.wraps(func)
    def wrapper():
        if func.__name__ not in cache:  # 如果函数名未在缓存中，则计算结果并缓存
            cache[func.__name__] = func()
        return cache[func.__name__]  # 返回缓存中的结果

    return wrapper