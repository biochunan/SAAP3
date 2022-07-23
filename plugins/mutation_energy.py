"""
Evaluate to-proline mutation
"""
from collections import OrderedDict
import argparse
import logging
import re
import shutil
from pathlib import Path
from pprint import pformat
from typing import Union, List, Dict, Tuple

import Bio.PDB.Structure
import numpy as np
import pandas as pd
from Bio.Data import IUPACData
from Bio.PDB import PDBIO
from Bio.PDB import PDBParser
from Bio.PDB import Select, Chain

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s {%(pathname)s:%(lineno)d} [%(levelname)s] %(name)s - %(message)s [%(threadName)s]',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(".proline")

AA_3to1: Dict[str, str] = {k.upper(): v for k, v in IUPACData.protein_letters_3to1.items()}
AA_3to1_EXT: Dict[str, str] = {k.upper(): v for k, v in IUPACData.protein_letters_3to1_extended.items()}

# ============================== Configuration ==============================
InfoString = "Checking if this is a mutation to proline"
PLUGIN_DATA_DIR = Path("/Users/chunan/UCL/scripts/SAAP3/plugins/data")
GlyTorsionDensityMap = PLUGIN_DATA_DIR.joinpath("heatMap_pc25res1.8R0.3_Gly_sm6_energyMatrix.txt")
ProTorsionDensityMap = PLUGIN_DATA_DIR.joinpath("heatMap_pc25res1.8R0.3_Pro_sm6_energyMatrix.txt")
ElseTorsionDensityMap = PLUGIN_DATA_DIR.joinpath("heatMap_pc25res1.8R0.3_Else_sm6_energyMatrix.txt")
GlyThreshold = 0.35
ProThreshold = 0.53
ElseThreshold = 1.5


# ============================== Functions ==============================
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdbfile', nargs=1, type=str, help="Path to pdb file")
    parser.add_argument('residue', nargs=1, type=str, help="[chain]resi[insertion_code] e.g. 30, L30, L30A")
    parser.add_argument('newaa', nargs=1, type=str,
                        help="mutant amino acid, must be 3- or 1-letter code, e.g. Pro or P")
    parser.add_argument('--force', action="store_true")

    args = parser.parse_args()
    residue = args.residue[0]
    new_aa = args.newaa[0]
    pdbfile = args.pdbfile[0]
    force = args.force

    # processing
    new_aa = new_aa.upper() if len(new_aa) == 1 else AA_3to1_EXT[new_aa.upper()]
    pdbfile = Path(pdbfile)

    return residue, new_aa, pdbfile, force


# TODO [chunan]: check if results are cached
def check_cache(program: str, pdbfile: str, resid: str, newres: str) -> Union[Dict, str]:
    cache_file = f"{pdbfile}_{resid}_{newres}"
    pass


# [x] [chunan]: parse residue info
def parse_residue_identifier(residue_identifier: str) -> Tuple[str, int, str]:
    """
    Args:
        residue_identifier: (str) e.g. 30, L30, L30A
    Returns:
        chain: (str) chain id
        resi: (int) residue index
        ins: (str) residue index insertion code
    """
    chain, resi, ins = None, None, None
    if "." in residue_identifier:
        chain, resi, ins = re.search(r"([0-9]+)\.(\d+)([A-Za-z]*)",
                                     residue_identifier).groups()
    else:
        chain, resi, ins = re.search(r"([A-Za-z]+)(\d+)(\w*)",
                                     residue_identifier).groups()
    resi = int(resi)
    try:
        assert chain is not None and resi is not None and ins is not None
    except AssertionError:
        raise ValueError(f"Parsing residue name error.")

    return chain, resi, ins


def read_pdb(pdb_fp: Path) -> Bio.PDB.Structure.Structure:
    pdbid, pdbfp = pdb_fp.stem, pdb_fp
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(id=pdbid, file=pdbfp)
    return struct


class ExtractChain(Select):
    def __init__(self, chain_id_to_extract: str):
        self.chain_id_to_extract = chain_id_to_extract

    def accept_chain(self, chain):
        """
        Returns:
            (bool): True if the input chain id mathced with chain_id_to_extract
        """
        return chain.get_id() in self.chain_id_to_extract


def get_chain_from_pdb(pdb_fp: Path, chain: str, out_dir: Path) -> Path:
    """
    Args:
        pdb_fp: input pdb file path
        chain: (str) chain id
        out_dir: (Path) directory to output chain file

    Returns:
        chain_fp: (Path) output file path to write the extracted chain corodinate file
    """

    chain_fp = out_dir.joinpath(f"{chain}.pdb")

    if chain == "":
        shutil.copy(src=pdb_fp, dst=chain_fp)
    else:
        structure = read_pdb(pdb_fp=pdb_fp)
        io = PDBIO()
        io.set_structure(structure)
        selection = ExtractChain(chain)
        io.save(chain_fp, select=selection)

    return chain_fp


def get_native_residue_name(struct_df: pd.DataFrame, chain_id: str, residue_index: int, insertion_code: str) -> str:
    """

    Args:
        struct_df: (pd.DataFrame) structure DataFrame
        chain_id: (str) chain id
        residue_index: (int) residue index e.g. 24
        insertion_code: (str) insertion code e.g. "A"

    Returns:
        resname: (str) native residue name 3-letter code, e.g. PRO
    """
    native_resname = struct_df[(struct_df.chain == chain_id) &
                               (struct_df.resi == residue_index) &
                               (struct_df.alt == insertion_code)].drop_duplicates("node_id").resn.values[0]

    if not native_resname:
        raise ValueError(f"{chain_id}{residue_index}{insertion_code} not found.")

    return native_resname


def init_struct_dict():
    return dict(
        node_id=[], chain=[],  # chain
        resi=[], alt=[], resn=[],  # residue
        atom=[], element=[],  # atom
        x=[], y=[], z=[]  # coordinate
    )


def unpack_chain(chain_obj: Bio.PDB.Chain.Chain,
                 retain_hetatm: bool = False,
                 retain_water: bool = False,
                 retain_b_factor: bool = False) \
        -> Dict[str, List[Union[str, int, float]]]:
    """
    Unpack a Biopython Chian object into a dictionary containing

    node_id: (List[int]) zero-based residue index
    chain: (List[str]) chain id
    resi:  (List[int]) residue index
    alt: (List[str]) insertion code
    resn: (List[str]) residue one-letter code
    atom: (List[str]) atom name
    element: (List[str]) atom element type, remove Hydrogen atoms from df
    x, y, z: (List[float]) coordinates

    Args:
        chain_obj: (Bio.PDB.Chain.Chain)
        retain_water: (bool) if True, add WATER ("W") atoms to DataFrame
        retain_hetatm: (bool) if True, add HETATM ("H_*") atoms to DataFrame
        retain_b_factor: (bool) if True, add `b_factor` column to DataFrame

    Returns:
        d: Dict[str, List[Union[str, int, float]]]
    """
    n = -1  # in case of HETATM residue
    chain_id = chain_obj.id
    d = init_struct_dict()
    if retain_b_factor:
        d["b_factor"] = []

    def _add_atom(d: Dict, atm: Bio.PDB.Atom.Atom,
                  chain_id: str, resi: str, alt: str, resn: str):
        d["chain"].append(chain_id)
        d["resi"].append(int(resi))
        d["alt"].append(alt)
        d["resn"].append(resn)
        d["atom"].append(atm.id)
        d["element"].append(atm.element)
        # coord
        x, y, z = atm.coord
        d["x"].append(float(x))
        d["y"].append(float(y))
        d["z"].append(float(z))
        # b_factor
        if retain_b_factor:
            d["b_factor"].append(atm.get_bfactor())

    for _, res in enumerate(chain_obj):
        # residue info
        het, resi, alt = res.id
        alt = "" if alt == " " else alt
        if het == " ":  # amino acid
            n += 1
            resn = AA_3to1[res.resname]
            atms = res.child_list  # atms = [a.id for a in res.child_list]
            for a in atms:
                d["node_id"].append(n)
                _add_atom(d=d, atm=a, chain_id=chain_id, resi=resi, alt=alt, resn=resn)
        elif het == "W" and retain_water:  # water solvent
            resn = res.resname
            atms = res.child_list  # atms = [a.id for a in res.child_list]
            for a in atms:
                d["node_id"].append(None)
                _add_atom(d=d, atm=a, chain_id=chain_id, resi=resi, alt=alt, resn=resn)
        elif het.startswith("H_") and retain_hetatm:  # HETATM
            resn = f"H_{res.resname}"
            atms = res.child_list  # atms = [a.id for a in res.child_list]
            for a in atms:
                d["node_id"].append(None)
                _add_atom(d=d, atm=a, chain_id=chain_id, resi=resi, alt=alt, resn=resn)
    return d


def chain2df(chain_obj: Bio.PDB.Chain.Chain, **kwargs) -> pd.DataFrame:
    """
    Turn structure chain into DataFrame, iterate over residues
    Wrapper of function unpack_chain, to accept a struct

    Args:
        chain_obj: (chain: Bio.PDB.Chain.Chain)

    Returns:
        df: (pd.DataFrame) DataFrame
    """
    # use default kwargs if not specified
    # d = dict(node_id=[], chain=[], resi=[], alt=[], resn=[], atom=[], element=[], x=[], y=[], z=[])
    retain_hetatm = kwargs.get("retain_hetatm", False)
    retain_water = kwargs.get("retain_water", False)
    retain_b_factor = kwargs.get("retain_b_factor", False)

    # unpack chain
    d = unpack_chain(chain_obj=chain_obj,
                     retain_hetatm=retain_hetatm,
                     retain_water=retain_water,
                     retain_b_factor=retain_b_factor)

    df = pd.DataFrame(d)
    # curate column data type
    df["node_id"] = df.node_id.astype("Int64")

    return df


def calc_dihedral_np(a_coords: np.ndarray,
                     b_coords: np.ndarray,
                     c_coords: np.ndarray,
                     d_coords: np.ndarray,
                     convert_to_degree: bool = True):
    """Rewrite of calc_dihedral_torch in numpy """
    b1 = a_coords - b_coords
    b2 = b_coords - c_coords
    b3 = c_coords - d_coords

    n1 = np.cross(b1, b2)
    n1 = n1 / np.linalg.norm(n1, axis=-1, keepdims=True)

    n2 = np.cross(b2, b3)
    n2 = n2 / np.linalg.norm(n2, axis=-1, keepdims=True)

    m1 = np.cross(n1, b2 / np.linalg.norm(b2, axis=-1, keepdims=True))

    dihedral = np.arctan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))

    if convert_to_degree:
        dihedral = dihedral * 180 / np.pi

    return dihedral


def get_phi_psi(struct_df: pd.DataFrame,
                chain_id: str,
                residue_index: int,
                insertion_code: str) -> Tuple[np.ndarray, np.ndarray]:
    identifier = f"{chain_id}{residue_index}{insertion_code}"

    # get node id
    node_id = None
    node_id = struct_df[(struct_df.chain == chain_id) &
                        (struct_df.resi == residue_index) &
                        (struct_df.alt == insertion_code)].drop_duplicates("node_id").node_id.values[0]
    if node_id is None:
        logger.error(f"Did not find residue {identifier}.")
        raise ValueError(f"Did not find residue {identifier}.")

    # get residues i-1, i, i+1
    ri = struct_df[struct_df.node_id == node_id]
    ra = struct_df[struct_df.node_id == node_id - 1]
    rb = struct_df[struct_df.node_id == node_id + 1]

    preceding_res_exist = True
    if ra.shape[0] == 0:
        logger.warning(f"Did not find preceding residue for {identifier}")
        preceding_res_exist = False

    succeeding_res_exist = True
    if rb.shape[0] == 0:
        logger.warning(f"Did not find succeeding residue for {identifier}")
        succeeding_res_exist = False

    # get coord
    ri_N = ri[ri.atom == "N"][["x", "y", "z"]].to_numpy()
    ri_CA = ri[ri.atom == "CA"][["x", "y", "z"]].to_numpy()
    ri_C = ri[ri.atom == "C"][["x", "y", "z"]].to_numpy()

    # phi and psi
    phi, psi = None, None
    if preceding_res_exist:
        # phi: i-1.C = i.N  - i.CA - i.C
        ra_C = ra[ra.atom == "C"][["x", "y", "z"]].to_numpy()
        phi = calc_dihedral_np(ra_C, ri_N, ri_CA, ri_C)

    if succeeding_res_exist:
        # psi: i.N   - i.CA - i.C  - i+1.N
        rb_N = rb[rb.atom == "N"][["x", "y", "z"]].to_numpy()
        psi = calc_dihedral_np(ri_N, ri_CA, ri_C, rb_N)

    return phi, psi


def read_torsion_density_map(density_map_file: Path) -> np.ndarray:
    arr = []
    with open(density_map_file, "r") as f:
        f.readline()
        for l in f:
            arr.append([float(i) for i in l.split(",")[1:]])
    arr = np.array(arr)
    return arr


def round_and_limit(val: float) -> int:
    """
    Args:
        val: (float) dihedral angle phi or psi

    Returns:
        val: (int) converted to bin index -180, -179, ..., -1, 0, 1, 2, ..., 178, 179
            左闭右开 [-180., -179.), [-179., -178.) ... [-1., 0.), [0, 1), ..., [178, 179), [179, 180)
    """
    val = int(np.floor(val - 0.5))  # 179.5 => 179

    # correct boundary
    if val < -180:
        val = -180
    elif val > 179:
        val = 179

    return val


def check_mutation_phi_psi_energy(phi: float,
                                  psi: float,
                                  native_residue_name: str,
                                  mutant_residue_name: str,
                                  gly_torsion_density_map_fp: Path,
                                  pro_torsion_density_map_fp: Path,
                                  else_torsion_density_map_fp: Path):
    """
    Args:
        phi: (float) dihedral angle
        psi: (float) dihedral angle
        native_residue_name: (str) native residue name 1-letter code
        gly_torsion_density_map_fp:
        pro_torsion_density_map_fp:
        else_torsion_density_map_fp:

    Returns:

    """
    native_residue_name = native_residue_name.upper()
    assert len(native_residue_name) == 1 and native_residue_name != "P"

    # [x] [chunan]: read_torsion_density_map
    gly_mat = read_torsion_density_map(
        gly_torsion_density_map_fp)  # shape [360, 360], -180, -179, ..., -1, 0, 1, 2, ..., 178, 179
    pro_mat = read_torsion_density_map(pro_torsion_density_map_fp)
    else_mat = read_torsion_density_map(else_torsion_density_map_fp)

    # [x] [chunan]: to grid indexing
    phi, psi = round_and_limit(phi), round_and_limit(psi)  # to int, also arr indexing

    # [x] [chunan]: correct the indexing
    gly_energy = gly_mat[phi + 180, psi + 180]  # e.g. both phi and psi -180 => [0, 0]
    pro_energy = pro_mat[phi + 180, psi + 180]  # e.g. both phi and psi  180 => [360, 360]
    else_energy = else_mat[phi + 180, psi + 180]

    # Check native residue
    from_glycine, bad_nat, nat_energy, nat_threshold = False, True, None, None
    if native_residue_name == "G":
        # if native is glycine
        from_glycine = True
        nat_energy = gly_energy
        nat_threshold = GlyThreshold
    else:
        # if native is other aa type
        nat_energy = else_energy
        nat_threshold = ElseThreshold
    if nat_energy < nat_threshold:
        bad_nat = False  # native residue energy in the allowed region

    # Check mutant residue
    to_proline, bad_mut, mut_energy, mut_threshold = False, True, None, None
    if mutant_residue_name == "P":
        # if mutant is proline
        to_proline = True
        mut_energy = pro_energy
        mut_threshold = ProThreshold
    else:
        # if mutant is non-proline
        mut_energy = else_energy
        mut_threshold = ElseThreshold
    if mut_energy < mut_threshold:
        bad_mut = False  # Native Gly in the allowed region

    return from_glycine, bad_nat, nat_energy, nat_threshold, to_proline, bad_mut, mut_energy, mut_threshold


def usage():
    usage_str = """
proline.pl V3.2 (c) 2011-2020, UCL, Prof. Andrew C.R. Martin

Usage: proline.pl residue newaa pdbfile [--force]
    
    residue  [chain]resnum[insert] e.g. 30 or L30 or L30A 
    newaa    newaa maybe 3-letter or 1-letter code
    pdbfile  path to pdb file 
    --force  Force calculation even if results are cached

Does proline calculations for the SAAP server.\n

"""
    print(usage_str)
    exit(1)


# ============================== Main ==============================
if __name__ == "__main__":
    # [x] [chunan]: get command line input
    residue_identifier, new_aa, struct_fp, force_cal = cli()
    struct_id = struct_fp.stem

    json_str = ""
    # TODO [chunan]: see if results are cached
    # json = check_cache(program="Proline", pdbfile=pdb_fp, resid=residue_identifier, newres=new_aa)

    if force_cal:
        json_str = ""

    if json_str != "":
        print(json_str)
        exit(0)

    # [x] [chunan]: parse residue identifier
    chain, residue_index, insertion = parse_residue_identifier(residue_identifier)

    # [x] [chunan]: parse pdb chain
    chain_df = chain2df(chain_obj=read_pdb(pdb_fp=struct_fp)[0][chain])
    # chain_file = get_chain_from_pdb(pdb_fp=pdb_fp, chain=chain, out_dir=Path.cwd())
    native_residue_name = get_native_residue_name(struct_df=chain_df,
                                                  chain_id=chain,
                                                  residue_index=residue_index,
                                                  insertion_code=insertion)
    phi, psi = get_phi_psi(struct_df=chain_df,
                           chain_id=chain,
                           residue_index=residue_index,
                           insertion_code=insertion)

    nat_energy, pro_energy, nat_threshold = None, None, None
    nat_result, mut_result = "okay", "okay"

    # assert both phi and psi are real number
    if phi is None or psi is None:
        logger.error("Terminal residue - analysis not performed as Phi/Psi angles could not be calculated")
        exit(1)

    # check energy region for native and mutate aa_type
    from_glycine, bad_nat, nat_energy, nat_threshold, to_proline, bad_mut, mut_energy, mut_threshold = \
        check_mutation_phi_psi_energy(phi=float(phi),
                                      psi=float(psi),
                                      native_residue_name=native_residue_name,
                                      mutant_residue_name=new_aa,
                                      gly_torsion_density_map_fp=GlyTorsionDensityMap,
                                      pro_torsion_density_map_fp=ProTorsionDensityMap,
                                      else_torsion_density_map_fp=ElseTorsionDensityMap)

    # mutation result
    if bad_nat:
        # report native residue energy
        nat_result = "bad"
    else:
        if bad_mut:
            # report mutant residue energy
            mut_result = "bad"

    # from-glycine mutation: mutate GLY to other aa_type

    # report mutate to proline results
    result = OrderedDict({
        "struct_id": struct_id,
        "residue_identifier": residue_identifier,
        "phi": float(phi),
        "psi": float(psi),
        "from_glycine": from_glycine,
        "to_proline": to_proline,
        "native_residue": native_residue_name,
        "native_result": nat_result,
        "native_energy": nat_energy,
        "native_thr": nat_threshold,
        "mutant_residue": new_aa,
        "mutant_result": mut_result,
        "mutant_energy": mut_energy,
        "mutant_thr": mut_threshold,
    })

    print("Results:\n"
          f"struct_id           {struct_id}\n"
          f"residue_identifier  {residue_identifier}\n"
          f"phi                 {float(phi)}\n"
          f"psi                 {float(psi)}\n"
          f"from_glycine        {from_glycine}\n"
          f"to_proline          {to_proline}\n"
          f"native_residue      {native_residue_name}\n"
          f"native_result       {nat_result}\n"
          f"native_energy       {nat_energy}\n"
          f"native_thr          {nat_threshold}\n"
          f"mutant_residue      {new_aa}\n"
          f"mutant_result       {mut_result}\n"
          f"mutant_energy       {mut_energy}\n"
          f"mutant_thr          {mut_threshold}\n")
