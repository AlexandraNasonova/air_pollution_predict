import os

def get_params(env_file_path):
    # __location__ = os.path.realpath(
    #     os.path.join(os.getcwd(), os.path.dirname(__file__)))
    # with open(os.path.join(__location__, env_file_path), 'r') as fh:
    #     vars_dict = dict(
    #         tuple(line.replace('\n', '').split('='))
    #         for line in fh.readlines() if not line.startswith('#')
    #     )
    # return vars_dict
    # __location__ = os.path.realpath(
    #     os.path.join(os.getcwd(), os.path.dirname(__file__)))
    with open(os.path.join(env_file_path), 'r') as fh:
        vars_dict = dict(
            tuple(line.replace('\n', '').split('='))
            for line in fh.readlines() if not line.startswith('#')
        )
    return vars_dict