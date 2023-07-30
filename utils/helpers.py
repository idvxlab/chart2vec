import functools

def memoize(func):
    # dictionary for caching results
    cache = {}  
    @functools.wraps(func)
    def wrapper():
        # if the function name is not in the cache, the result is computed and cached
        if func.__name__ not in cache:  # 如果函数名未在缓存中，则计算结果并缓存
            cache[func.__name__] = func()
        # returns cached results
        return cache[func.__name__]
    return wrapper