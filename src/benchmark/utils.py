import sys

def list_modules(snapshot, message):
    
    '''DEPRICATED: This functions works out which libraries are essential for the
    execution of the benchmarking tool'''
    
    current_modules = set(sys.modules.keys())
    new_modules = current_modules - snapshot
    print(f"{message} (newly loaded):")
    for module in sorted(new_modules):
        print(module)
    return current_modules