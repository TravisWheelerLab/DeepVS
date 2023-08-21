from plip.structure.preparation import PDBComplex
import re
import itertools

def get_ligand_data(pl_interaction):
	# Takes object returned by PDBComplex.analyze() method
	# returns graph of interactions 
    ligand_interaction_data = []

    for interaction in pl_interaction.all_itypes:
        i_type = re.search(r".*\.(\S+)\'\>$", str(type(interaction))).group(1)

        if i_type == 'hbond':
            if interaction.protisdon == True:
                interaction_record = [i_type+"_a", interaction.a.coords]
            else:
                interaction_record = [i_type+"_d", interaction.h.coords]

        if i_type == 'hydroph_interaction':
            interaction_record = [i_type, interaction.ligatom.coords] 

        if i_type == 'halogenbond':
            interaction_record = [i_type, interaction.don.orig_x.coords] 

        if i_type == 'pistack':
            interaction_record = [i_type, tuple(interaction.ligandring.center)]

        if i_type == 'saltbridge':
            if interaction.protispos:
                interaction_record = ['saltbridge_n', tuple(interaction.negative.center)]
            else:
                interaction_record = ['saltbridge_p', tuple(interaction.positive.center)]

        if i_type == 'pication':
            if interaction.protcharged:
                interaction_record = [i_type + '_r', tuple(interaction.ring.center)]
            else:
                interaction_record = [i_type + '_c', tuple(interaction.charge.center)]

        if i_type in ['metal_complex', 'waterbridge']: 
            continue

        ligand_interaction_data.append(interaction_record)

    return ligand_interaction_data 


def get_interaction_data(pdb_file):
    my_mol = PDBComplex()
    my_mol.load_pdb(pdb_file)
    my_mol.analyze()

    interaction_data = []

    for object_ids, pl_interaction in my_mol.interaction_sets.items():
        plip_profile = get_ligand_data(pl_interaction)
        interaction_data.extend(plip_profile)

    # Remove duplicates
    interaction_data = [x for x,_ in itertools.groupby(sorted(interaction_data))]
    return interaction_data

    