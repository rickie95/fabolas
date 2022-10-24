"""
    Performs experiments on SVMs trained on MNIST
"""

def obj_function():
    pass


if __name__ == "__main__":

    method = ''

    if method == 'random_search':
        import random_search
        random_search(obj_function, data)
        pass
    elif method == 'expected_improvement':
        pass
    elif method == 'entropy_search':
        pass
    elif method == 'fabolas':
        pass
