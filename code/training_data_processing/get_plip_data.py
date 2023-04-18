import os
import sys
import pickle
from glob import glob
import code.utils.plip_utils as plip_utils
import code.utils.data_processing_utils as data_utils
import code.utils.pdb_utils as pdb_utils
import string
import random

def get_plip_data(config: dict, id_batch: list, pdbbind_dir: str, data_dir: str=None, interaction_profile_dir: str=None, **kwargs) -> None:
    # Gather interaction data from all PDBBind complexes
    # Store interactions by location/type in interaction file  
    if interaction_profile_dir:
        ip_ft = interaction_profile_dir + config['interaction_profile_file_template'].split('/')[-1]
    else:
        ip_ft = data_dir + config['interaction_profile_file_template']
        interaction_profile_dir = '/'.join(ip_ft.split('/')[:-1]) + '/'

    if os.path.exists(interaction_profile_dir) == False:
        os.makedirs(interaction_profile_dir)

    if kwargs.get('skip'):
        id_batch = data_utils.trim_batch_ids(id_batch, ip_ft)

    batch_total = len(id_batch)
    pdbbind_dir = config['pdbbind_dir']

    ligand_ft = pdbbind_dir + "%s/%s_ligand.sdf"
    protein_ft = pdbbind_dir + "%s/%s_pocket.pdb"

    for target_count, target_id in enumerate(id_batch):
        protein_pdb = protein_ft % (target_id, target_id)
        ligand_sdf = ligand_ft % (target_id, target_id)

        # random string to be prepended to temporary files
        random_file_id = ''.join(random.choices(string.ascii_letters +
                             string.digits, k=7))

        ligand_pdb = "/tmp/%s_%s_ligand.pdb" % (random_file_id, target_id)
        complex_pdb = "/tmp/%s_%s_complex.pdb" % (random_file_id, target_id)
        ip_file = ip_ft % target_id

        # command line function to convert sdf file to pdb
        obabel_command = "obabel -isdf %s -opdb > %s"  % (ligand_sdf, ligand_pdb)
        os.system(obabel_command)

        # PLIP Requires a pdb complex file containing both protein and ligand
        # store ATOM/HETATM PDB lines here
        complex_pdb_content = pdb_utils.merge_pdbs(protein_pdb, ligand_pdb)
        
        with open(complex_pdb, 'w') as complex_out:
            complex_out.write(complex_pdb_content)

        interaction_data = plip_utils.get_interaction_data(complex_pdb) 

        # PLIP does not successfully fetch interaction data for all files
        # Only write interation profile if there is data available
        if len(interaction_data) > 0:
            pickle.dump(interaction_data, open(ip_file, 'wb'))

        for temp_file in glob('/tmp/*%s*' % random_file_id):
            os.remove(temp_file)

        print("PLIP Processed %s: %s of %s" % (target_id, target_count+1, batch_total))
