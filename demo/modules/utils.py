#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#


def noexcept(default_value):
    def process(function):
        def wrapper(*args, **kwargs):
            try:
                return_value = function(*args, **kwargs)
                return return_value
            except Exception:
                args[0].log.failure('')
                return default_value
        return wrapper
    return process
