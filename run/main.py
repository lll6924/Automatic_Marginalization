import click
import importlib
from jax import make_jaxpr, random
from jax.core import CallPrimitive
from numpyro.handlers import seed, trace
from utils import get_alphabetic_list

@click.command()
@click.option('--model', default='EightSchools', help = 'The Model to Perform Inference.')
def main(model):
    module = importlib.import_module('model')
    model = getattr(module, model)()
    key = random.PRNGKey(0)
    model_seeded = seed(model.model, key)
    model_trace = trace(model_seeded).get_trace(*model.args())

    # get variable names and ordering in NumPyro
    rvs = []
    for site in model_trace.values():
        if site["type"] == 'sample':
            rvs.append({'name':site['name']})

    jpr = make_jaxpr(model_seeded)(*model.args())#, **model.kwargs())
    eqns = jpr.eqns

    # loop over Jaxprs, identify the random variables
    is_rv = {}
    dep = {}
    car = {}
    n_consts = len(jpr.consts)
    n_in_vars = len(jpr.in_avals)
    for par in get_alphabetic_list(n_consts + n_in_vars):
        dep[par] = set()
        car[par] = set()
        is_rv[par] = False
    rv_id = 0
    expr_mapping = {}
    for e in eqns:
        if isinstance(e.primitive, CallPrimitive) and e.params['name'] == 'register':
            for var in e.invars:
                var = str(var)
                is_rv[var] = True
                car[var] = {var}
                rvs[rv_id]['expr'] = var
                expr_mapping[var] = rv_id
                rv_id += 1

        for var in e.outvars:
            is_rv[str(var)] = False
            var = str(var)
            dep[var] = set()
            for prev in e.invars:
                prev = str(prev)
                if prev in car.keys():
                    dep[var] = dep[var].union(car[prev])
            car[var] = dep[var]

    for k, w in dep.items():
        if is_rv[k]:
            rvs[expr_mapping[k]]['dep'] = []
            for d in w:
                if is_rv[d]:
                    rvs[expr_mapping[k]]['dep'].append(rvs[expr_mapping[d]]['name'])

    print(rvs)


    #for e in eqns:
    #    print(e.primitive, e.invars, e.outvars, e.params)

if __name__ == '__main__':
    main()