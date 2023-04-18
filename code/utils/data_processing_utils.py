from glob import glob

def trim_batch_ids(id_batch, file_template):
    completed_ids = [x.split('/')[-1].split('_')[0] for x in glob(file_template.replace('%s', '*'))]
    id_batch = [x for x in id_batch if x not in completed_ids]
    return id_batch

def get_output_paths(custom_output_dir, default_output_dir, default_file_template):
    if custom_output_dir:
        file_template = custom_output_dir + default_file_template.split('/')[-1]
        output_dir = custom_output_dir
    else:
        file_template =  default_output_dir + default_file_template
        output_dir = '/'.join(file_template.split('/')[:-1]) + '/'

    return output_dir, file_template
