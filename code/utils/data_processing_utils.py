from glob import glob
import os
import sys
import importlib.util


def load_class_from_file(file_path):
    class_name = file_path.split("/")[-1].split(".")[0]
    spec = importlib.util.spec_from_file_location(class_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def load_function_from_file(file_path):
    function_name = file_path.split("/")[-1].split(".")[0]
    spec = importlib.util.spec_from_file_location(
        os.path.basename(file_path), file_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def trim_batch_ids(id_batch, file_template):
    completed_ids = [
        x.split("/")[-1].split("_")[0] for x in glob(file_template.replace("%s", "*"))
    ]
    id_batch = [x for x in id_batch if x not in completed_ids]
    return id_batch


def get_output_paths(custom_output_dir, default_output_dir, default_file_template):
    if custom_output_dir:
        file_template = custom_output_dir + default_file_template.split("/")[-1]
        output_dir = custom_output_dir
    else:
        file_template = default_output_dir + default_file_template
        output_dir = "/".join(file_template.split("/")[:-1]) + "/"

    return output_dir, file_template

def interpolate_root(path, root_dir):
    if "%s" in path:
        if root_dir[-1] != "/":
            root_dir += "/"
        if path[0] == '/':
            path = path[1:]
        
        return path % root_dir
    else:
        return path
