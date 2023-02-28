from functools import reduce
from torch.nn.modules.module import _addindent
import sys



def summary(model,file=sys.stderr):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        if model is None:
            return 
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            if module is None:
                continue
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            # print("name is ", name)
            if p is not None:
                # print("parameter with shape ", p.shape)
                # print("parameter has dim ", p.dim())
                if p.dim()==0: #is just a scalar parameter
                    total_params+=1
                else:
                    total_params += reduce(lambda x, y: x * y, p.shape)
                # if(p.grad==None):
                #     print("p has no grad", name)
                # else:
                #     print("p has gradnorm ", name ,p.grad.norm() )

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        # if file is sys.stderr:
        if True:
            # main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
            main_str += ', {:,} params'.format(total_params)
            for name, p in model._parameters.items():
                if hasattr(p, 'data'):
                    main_str+= "\n\t{:f} is norm of {:s}".format(p.data.norm().item(), name)
                if hasattr(p, 'grad'):
                    if(p.grad==None):
                        main_str+="\n\tno grad"
                    else:                        
                        main_str+= "\n\t{:f} is grad norm of {:s}".format(p.grad.norm().item(), name)                        
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        print(string, file=file)
    return count
