from experiments.commands import shield_experiment, controller_experiment, cont_exp, combined_exp


BASE_DIR = './experiments/automated/'
models = ['dcdc_boost_converter']

for model in models:
    model_dir = BASE_DIR + model + '/'

    # results = controller_experiment(model_dir)
    # import ipdb; ipdb.set_trace()

    shield_results, wrapped_shield, sdata = shield_experiment(model_dir)
    print('\nshield res:')
    print(shield_results)
    print()

    # cont_results, v_tree = cont_exp(wrapped_shield, model_dir, sdata)
    # print('\ncontroller res:')
    # print(cont_results)

    # print('\nrunning combined experiment...')
    # try:
    #     comb_results = combined_exp(v_tree, wrapped_shield, sdata)
    # except:
    #     import ipdb; ipdb.set_trace()

    # print('\ncombined results:')
    # print(comb_results)
