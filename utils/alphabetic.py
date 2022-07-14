from string import ascii_lowercase
import itertools

# from https://stackoverflow.com/questions/29351492/how-to-make-a-continuous-alphabetic-list-python-from-a-z-then-from-aa-ab-ac-e
def iter_all_strings():
    for size in itertools.count(1):
        for s in itertools.product(ascii_lowercase, repeat=size):
            yield "".join(s)

def get_alphabetic_list(n):
    ret = []
    for s in itertools.islice(iter_all_strings(), n):
        ret.append(s)
    return ret