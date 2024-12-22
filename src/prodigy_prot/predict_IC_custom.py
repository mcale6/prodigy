#!/usr/bin/env python
import json
import os
from predict_IC import *
from modules.freesasa_tools import *
from modules.models import *
from modules.parsers import *
from modules.utils import *

#Formula: BSA = (ASAprotein1 + ASAprotein2) - ASAcomplex form elife paper, but in original code they actually use the relative SASA per reisdue
#ASAprotein1 = Accessible surface area of first protein alone
#ASAprotein2 = Accessible surface area of second protein alone
#ASAcomplex = Accessible surface area of the full complex
# Uses probe radius of 1.4Å (size of water molecule)

class CustomProdigy(Prodigy):
      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

      # overwrite the predict function to use dr-sasa or patched freeesa
      def predict(self, temp=None, distance_cutoff=5.5, acc_threshold=0.05):
        if temp is not None:
            self.temp = temp
        # Make selection dict from user option or PDB chains
        selection_dict = {}
        for igroup, group in enumerate(self.selection):
            chains = group.split(",")
            for chain in chains:
                if chain in selection_dict:
                    errmsg = (
                        "Selections must be disjoint sets: "
                        f"{chain} is repeated"
                    )
                    raise ValueError(errmsg)
                selection_dict[chain] = igroup

        # Contacts
        self.ic_network = calculate_ic(self.structure, d_cutoff=distance_cutoff, selection=selection_dict)
        self.bins = analyse_contacts(self.ic_network)
        # SASA
        self.asa_data, self.rsa_data, self.abs_diff_data = execute_freesasa_api2(self.structure)

        chain_sums_res = lambda d: {'total': sum(d.values()), 'per_chain': {chain: sum(v for (c, _, _,), v in d.items() if c == chain) for chain in {k[0] for k in d.keys()}}}
        chain_sums_atm= lambda d: {'total': sum(d.values()), 'per_chain': {chain: sum(v for (c, _, _, _), v in d.items() if c == chain) for chain in {k[0] for k in d.keys()}}}

        print(chain_sums_res(self.rsa_data))
        print(chain_sums_res(self.abs_diff_data))
        print(chain_sums_atm(self.asa_data))
        #print(self.asa_data)

        self.nis_a, self.nis_c, self.nis_p = analyse_nis(self.rsa_data, acc_threshold=acc_threshold)
        # Affinity Calculation
        self.ba_val = IC_NIS(
            self.bins["CC"],
            self.bins["AC"],
            self.bins["PP"],
            self.bins["AP"],
            self.nis_a,
            self.nis_c,
        )
        self.kd_val = dg_to_kd(self.ba_val, self.temp)
        return

def predict_binding_affinity(
    struct_path,
    selection=None,
    temperature=25.0,
    distance_cutoff=5.5,
    acc_threshold=0.05,
    save_results=False,
    output_dir=None,
    quiet=False):
    """ Predict binding affinity using the custom PRODIGY method in python.
    care the relative bsa is sometiems higher than 1. in the codebase of prodigy
    Temperature in Celsius for Kd predictio, Distance cutoff to calculate ICs, Accessibility threshold for BSA analysis
    """
    # Check and parse structure
    structure, n_chains, n_res = parse_structure(struct_path)
    print(f"[+] Parsed structure file {structure.id} ({n_chains} chains, {n_res} residues)")

    # Initialize Prodigy and predict
    prodigy = CustomProdigy(structure, selection, temperature)
    prodigy.predict(distance_cutoff=distance_cutoff, acc_threshold=acc_threshold)
    prodigy.print_prediction(quiet=quiet)
    results = prodigy.as_dict()

    if save_results:
        res_fname_json = os.path.basename(struct_path.replace(".pdb", "_ba_results.json"))
        res_fname_csv = os.path.basename(struct_path.replace(".pdb", "_sasa_atom_results.csv"))
        asa_csv_lines = "\n".join(["Chain,ResName,ResID,Atom,SASA"] + [f"{chain},{resname},{resid.strip()},{atom.strip()},{sasa:.3f}" for (chain, resname, resid, atom), sasa in prodigy.asa_data.items()])
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path_json = os.path.join(output_dir, res_fname_json)
            output_path_csv = os.path.join(output_dir, res_fname_csv)
        else:
            output_path_json = os.path.join(".", res_fname_json)
            output_path_csv = os.path.join(".", res_fname_csv)
        
        with open(output_path_json, "w") as json_file:
            json.dump(results, json_file, indent=4)
        with open(output_path_csv, "w") as f:
            f.write(asa_csv_lines)
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict binding affinity using the PRODIGY method.")
    
    parser.add_argument(
        "struct_path", 
        type=str, 
        help="Path to the input structure file."
    )
    parser.add_argument(
        "--selection", 
        type=str, 
        default=None, 
        help="Selection of atoms or residues (optional)."
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=25.0, 
        help="Temperature in Celsius for Kd prediction (default: 25.0)."
    )
    parser.add_argument(
        "--distance_cutoff", 
        type=float, 
        default=5.5, 
        help="Distance cutoff for interface contacts (default: 5.5 Å)."
    )
    parser.add_argument(
        "--acc_threshold", 
        type=float, 
        default=0.05, 
        help="Accessibility threshold for BSA analysis (default: 0.05)."
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save the prediction results to a file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the results (optional)."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Outputs only the predicted affinity value",
    )
    args = parser.parse_args()

    # Call the prediction function
    result = predict_binding_affinity(
        struct_path=args.struct_path,
        selection=args.selection,
        temperature=args.temperature,
        distance_cutoff=args.distance_cutoff,
        acc_threshold=args.acc_threshold,
        save_results=args.save_results,
        output_dir=args.output_dir,
        quiet=args.quiet
    )

    # Optionally print or save results
    print("Binding affinity prediction completed.")
    print(result)  # Customize based on the `CustomProdigy` output format.

if __name__ == "__main__":
    main()