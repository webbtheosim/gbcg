'''
File:         GBCG_spectral.py
Author:       Michael A. Webb
Description:  This file is used for implementation of a graph-based coarse-graining of molecules
              using the spectral grouping scheme. Various options are available and can be
              be observed by executing the script with the -h option. The code requires access
              to thee numpy, math, copy, os, and datetime modules along with addition modules
              distributed along with this code. All testing and execution was performed with
              Python 3.8.3 as distributed with anaconda.
Copyright (C) 2021  Michael A. Webb

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
'''

version=1

#==================================================================
# MODULUES
#==================================================================
import sys,argparse
from   math import *
import numpy as np
import time
import aux.pdb_parser as pdb
from   aux.massesDb import *
import copy,os
import datetime
import scipy.sparse.linalg as sp_linalg
import scipy

#==================================================================
# CONSTANTS AND CONVERSIONS
#==================================================================
mlight = 3.500
pdbdir= "pdb_files/"
mapdir= "map_files/"
xyzdir= "xyz_files/"
lmpdir= "lammpstrj_files/"

#==================================================================
#  AUX: create_parser
#==================================================================
def create_parser():

  parser = argparse.ArgumentParser(description='Generates a graph-based coarse-grained (GBCG) particle mapping based on a      specified degree of connectivity, i.e., the spectral grouping scheme.\
          Details of the formalism can be found in the publication: Michael A. Webb, Jean-Yves Delannoy, and Juan J. de Pablo, "Graph-based approach to Systematic Molecular Coarse-graining," J. Comp. Theory Comput., DOI: 10.1021/acs.jctc.8b00920. See below for various available options.')

  # ARGUMENTS

  parser.add_argument('-pdb'     ,dest='pdbfile',default=None,
                      help = 'Name for the reference molecule pdb file. The pdb file must have connectivity information. It can additionally provide desired atom type labels as the last column, and atom priorities can be listed in the occupancy field. (default: None)')

  parser.add_argument('-niter',dest='niter',default=1,help = 'Number of coarsening iterations. (default: 1) ')

  parser.add_argument('-weights'     ,dest='weights',default=None,
          help = 'Style of vertex weights to be used in adjacency matrix. Available options include "mass" and "diff"  (default = None)')

  parser.add_argument('-max_size'     ,dest='max_size',default=None,
                      help = 'Maximum mass of a CG bead. (default = None)')

  parser.add_argument('-similar'     ,dest='simrat',default=1.0,
          help = 'Threshold ratio for evaluating similarity among CG atom types. If the overlap in constituents for two CG atoms types exceeds this number, they will be treated as a single type. (default = 1.0)')

  parser.add_argument('-typing'     ,dest='typing',default="all",
                      help = 'Typing method for CG atom types. Option "all" will use all of the contituent atom types. Option "heavy" will only use the heavy atoms for the CG typing. (default = "all")')

  parser.add_argument('-mmap'     ,dest='mmap',default=None,
                      help = 'Supplies file name for map of the atom type masses. If none supplied, then the mass is guessed from the atom name extracted from the pdb file. (default: none)')

  parser.add_argument('-pmap'     ,dest='pmap',default=None,
          help = 'Supplies file name for a priority map of the atom types. If none supplied, then priority will be assumed from the mass. If ''pdb'', then priority will be taken from the occupancy record in the pdb file (default: none)')

  parser.add_argument('-qmap'     ,dest='qmap',default=None,
          help = 'Supplies file name for a charge map of the atom types. If none supplied, then charge will be assumed to be zero. If ''pdb'', then charge will be taken from either the tempFactor or the last record in the pdb file (default: none)')
  return parser


#==================================================================
#  CLASS: OPTIONS
#==================================================================
class OPTIONS(object):
  def __init__(self,args):
    self.style  = args.weights
    self.niter  = int(args.niter)
    self.simrat = float(args.simrat)
    self.typing = args.typing
    if args.max_size is not None:
      self.max_size = float(args.max_size)
    else:
      self.max_size = float('inf')

#==================================================================
#  AUX: convert_args
#==================================================================
def convert_args(args):

  # SET FILES
  files = {}
  files['pdb']     = args.pdbfile
  files['mmap']    = args.mmap
  files['pmap']    = args.pmap
  files['qmap']    = args.qmap
  files['summary'] = open("summary.txt","w")

  # SET ARGUMENTS TO OPTIONS STRUCTURE
  options            = OPTIONS(args)

  # PRINT OUT CONFIRMATION OF PARAMETERS
  print_status("Processing input")
  print("# Structure file: ", files['pdb'])
  print("# Threshold size set to {} ...".format(options.max_size))
  print("# Number of coarsening rounds set to {} ...".format(options.niter))

  return (files,options)

#==================================================================
#  AUX: eigenvalue
#==================================================================
def eigenvalue(A, v):
    Av = A.dot(v)
    return v.dot(Av)

#==================================================================
#  AUX: power_iteration
#==================================================================
def power_iteration(A):
    n, d = A.shape
    v = np.ones(d) / np.sqrt(d)
    ev = eigenvalue(A, v)
    cnt=0
    while True:
        cnt+=1
        Av = A.dot(v)
        v_new = Av / np.linalg.norm(Av)
        ev_new = eigenvalue(A, v_new)
        if np.abs(ev - ev_new) <= 1.e-15:
            print("Power iteration convergence to 1e-15 achieved in {} iterations".format(cnt))
            break
        if cnt > 200000:
            print("Power iteration did not converge to <1e-15. Exiting with convergence of {:>10.5e}".format(np.abs(ev - ev_new)))
            break
        v = v_new
        ev = ev_new

    return ev_new, v_new

#==================================================================
#  AUX: print_status
#==================================================================
def print_status(status):
    print("\n********************************************************")
    print("# {}...".format(status))
    print("********************************************************")
    return

#==================================================================
#  AUX: is_number
#==================================================================
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

#==================================================================
#  AUX: make_directories
#==================================================================
def make_directories(dirs):
    for diri in dirs:
        if not os.path.exists(diri):
            os.makedirs(diri)
    return

#==================================================================
#  AUX: approximate_sigma
#==================================================================
def approximate_sigma(node,supp,atoms):
    nodeId = atmMap[node]
    rmax = atoms['coords'][nodeId].copy()
    rmin = atoms['coords'][nodeId].copy()
    for i in supp:
        iId = atmMap[i]
        ri = atoms['coords'][iId].copy()
        rmax[ri>rmax] = ri[ri>rmax]
        rmin[ri<rmin] = ri[ri<rmin]
    rmax += 2.5
    rmin -= 2.5
    dr   = rmax - rmin
    ravg = np.mean(dr)
    return ravg*0.5

#==================================================================
#  AUX: compute_angle
#==================================================================
def compute_angle(ri,rj,rk):
    rji = ri - rj
    rjk = rk - rj
    cost= np.dot(rji,rjk) / np.linalg.norm(rji) / np.linalg.norm(rjk)
    theta = np.arccos(cost) * 180.0 / np.pi
    return theta

#==================================================================
#  AUX: wrap_into_box
#==================================================================
def wrap_into_box(r,box):
    # shift origin to 0 0 0 to do wrapping
    L = np.array([x[1]-x[0] for x in box])
    r -= box[:,0]
    shift = np.array([floor(ri/Li) for ri,Li in zip(r,L)])
    r -= shift*L
    # shift back
    r += box[:,0]
    return r

#==================================================================
#  AUX: open_files
#==================================================================
def open_files(options,mols):
  # make the directories to contain coordinate files
  fpdb    = [[] for mol in mols]
  fmap    = []
  fout    = open("summary.txt","w")

  make_directories([pdbdir,mapdir])
  for i, moli in enumerate(mols):
    fname_pdb = pdbdir + "CG.mol_" + repr(i) + ".0.pdb"
    fpdb[i].append(open(fname_pdb,"w"))
    for iIter in range(options.niter):
        fname_pdb = pdbdir + "mol_" + repr(i) + "." + repr(iIter+1) +  ".pdb"
        fpdb[i].append(open(fname_pdb,"w"))

  fname_map = mapdir + "CG.map"
  fmap.append(open(fname_map,"w"))
  for iIter in range(options.niter):
        fname_map = mapdir + "iter." + repr(iIter+1) +  ".map"
        fmap.append(open(fname_map,"w"))

  return fpdb,fmap,fout

#==================================================================
# AUX: write_data_file
#==================================================================
def write_data_file(ftyp,atoms,CGmols,box,nOfType,CGmap):

    # acquire system information
    nCgType  = len(nOfType)
    nBonType = 0
    nAngType = 0
    nDihType = 0

    # total number of atoms
    natm = sum([len(i) for i in CGmols])

    # form adjacency matrix and bonds list and get size info
    adjmat = np.zeros([natm+1,natm+1])
    bonds    = []
    bond2typ = []
    btypes = []
    bavg   = [] # stores average bond length and number of occurences per bond type
    id2typ = {}
    coords = {}
    savg   = [[0.,0.] for i in range(nCgType)]
    for mol in CGmols:
        for i in sorted(mol.keys()):
            bead = mol[i]
            radius = approximate_sigma(i,bead['in'],atoms)
            idi = bead['id']
            ityp= bead['type']
            savg[ityp][0] += radius
            savg[ityp][1] += 1.0
            coords[idi] = bead['coords'][:]
            ri  = bead['coords'][:]
            id2typ[idi] = ityp
            for j in bead['adj']:
                idj = mol[j]['id']
                jtyp= mol[j]['type']
                rj  = mol[j]['coords'][:]
                adjmat[idi,idj] = 1
                adjmat[idj,idi] = 1
                if ityp < jtyp:
                    btype = (ityp,jtyp)
                else:
                    btype = (jtyp,ityp)
                if btype not in btypes:
                    btypes.append(btype)
                    bavg.append([0.0,0.0])
                if (idi < idj):
                    bonds.append((idi,idj))
                    dr   = ri - rj
                    bond2typ.append(btypes.index(btype))
                    bavg[btypes.index(btype)][0] += np.linalg.norm(dr)
                    bavg[btypes.index(btype)][1] += 1.0
    nbonds  = len(bonds)

    # angles
    angles  = []
    ang2typ = []
    atypes  = []
    aavg    = []
    for (i,j) in bonds:
        (potkj,) = np.nonzero(adjmat[j,:])
        (potki,) = np.nonzero(adjmat[i,:])
        kjs = [k for k in potkj if k != i]
        kis = [k for k in potki if k != j]
        ityp = id2typ[i]
        jtyp = id2typ[j]
        ri = coords[i][:]
        rj = coords[j][:]

        # check connections from j --> k (i,j,k) or (k,j,i)
        for k in kjs:
            rk = coords[k][:]
            ktyp = id2typ[k]
            if i < k:
                ang = (i,j,k)
            else:
                ang = (k,j,i)
            if ang not in angles:
                if ityp < ktyp:
                    atype = (ityp,jtyp,ktyp)
                else:
                    atype=(ktyp,jtyp,ityp)
                if atype not in atypes:
                    atypes.append(atype)
                    aavg.append([0.,0.])
                theta = compute_angle(ri,rj,rk)
                angles.append(ang)
                ang2typ.append(atypes.index(atype))
                aavg[atypes.index(atype)][0] += theta
                aavg[atypes.index(atype)][1] += 1.0

        # check connections from i --> k (j,i,k) or (k,i,j)
        for k in kis:
            rk = coords[k][:]
            ktyp = id2typ[k]
            if j < k:
                ang = (j,i,k)
            else:
                ang = (k,i,j)
            if ang not in angles:
                if jtyp < ktyp:
                    atype=(jtyp,ityp,ktyp)
                else:
                    atype=(ktyp,ityp,jtyp)
                if atype not in atypes:
                    atypes.append(atype)
                    aavg.append([0.,0.])
                theta = compute_angle(ri,rj,rk)
                angles.append(ang)
                ang2typ.append(atypes.index(atype))
                aavg[atypes.index(atype)][0] += theta
                aavg[atypes.index(atype)][1] += 1.0

    nangles = len(angles)
    deg2 = np.dot(adjmat[:,:],adjmat[:,:])
    print("nangles = ", nangles, (deg2.sum() - deg2.trace())/2.0)

    # dihedrals
    # XXX right now excludes connections that go to any of the constituent atom angles
    # may need to add impropers
    dihedrals = []
    dih2typ   = []
    dtypes = []
    for (i,j,k) in angles:
        (potlk,) = np.nonzero(adjmat[k,:])
        (potli,) = np.nonzero(adjmat[i,:])
        lks = [l for l in potlk if l not in [i,j]]
        lis = [l for l in potli if l not in [j,k]]
        ityp = id2typ[i]
        jtyp = id2typ[j]
        ktyp = id2typ[k]

        # check (i,j,k,l) or (l,k,j,i)
        for l in lks:
            ltyp = id2typ[l]
            if i < l:
                dih = (i,j,k,l)
            else:
                dih = (l,k,j,i)
            if dih not in dihedrals:
                if ityp < ltyp:
                    dtype=(ityp,jtyp,ktyp,ltyp)
                else:
                    dtype=(ltyp,ktyp,jtyp,ityp)
                if dtype not in dtypes:
                    dtypes.append(dtype)
                dihedrals.append(dih)
                dih2typ.append(dtypes.index(dtype))

        # check (l,i,j,k) or (k,j,i,l)
        for l in lis:
            ltyp = id2typ[l]
            if l < k:
                dih = (l,i,j,k)
            else:
                dih = (k,j,i,l)
            if dih not in dihedrals:
                if ltyp < ktyp:
                    dtype=(ltyp,ityp,jtyp,ktyp)
                else:
                    dtype=(ktyp,jtyp,ityp,ltyp)
                if dtype not in dtypes:
                    dtypes.append(dtype)
                dihedrals.append(dih)
                dih2typ.append(dtypes.index(dtype))

    deg3 = np.dot(deg2,adjmat)
    ndihedrals = len(dihedrals)
    print("ndihedrals = ", ndihedrals, (deg3.sum() - deg3.trace())/2.0)

    # write out the data file
    nBonType = len(btypes)
    nAngType = len(atypes)
    nDihType = len(dtypes)
    fid = open("sys.cg.data","w")
    fid.write("LAMMPS data file via GBCG Mapping, version {} {}\n\n".format(version,str(datetime.date.today())))
    fid.write("{} atoms\n".format(natm))
    fid.write("{} atom types\n".format(nCgType))
    fid.write("{} bonds\n".format(nbonds))
    fid.write("{} bond types\n".format(nBonType))
    fid.write("{} angles\n".format(nangles))
    fid.write("{} angle types\n".format(nAngType))
    fid.write("{} dihedrals\n".format(ndihedrals))
    fid.write("{} dihedral types\n\n".format(nDihType))

    # box
    fid.write("{} {} xlo xhi\n".format(box[0,0],box[0,1]))
    fid.write("{} {} ylo yhi\n".format(box[1,0],box[1,1]))
    fid.write("{} {} zlo zhi\n\n".format(box[2,0],box[2,1]))

    # Masses
    fid.write("Masses\n\n")
    for i,CGtype in CGmap.items():
        fid.write("{} {:>10.4f}\n".format(i+1,CGtype['mass'])) # add one from internal typing

    # Pair Coeffs
    fid.write("\nPair Coeffs\n\n")
    for i,CGtype in CGmap.items():
        savg[i][0] /= savg[i][1]
        fid.write("{:<5d} {:>10.4f} {:>10.4f}\n".format(i+1,0.1,savg[i][0]))

    # Bond Coeffs
    fid.write("\nBond Coeffs\n\n")
    ftyp.write("\n~~~~~BOND TYPES~~~~~\n")
    for i,btype in enumerate(btypes):
        bavg[i][0] /= bavg[i][1]
        fid.write("{} {:>8.3f} {:>8.3f}\n".format(i+1,100.0,bavg[i][0]))
        ftyp.write("Type {}: ({})--({})\n".format(i+1,btype[0]+1,btype[1]+1))

    # Angle Coeffs
    ftyp.write("\n~~~~~ANGLE TYPES~~~~~\n")
    fid.write("\nAngle Coeffs\n\n")
    for i,atype in enumerate(atypes):
        aavg[i][0] /= aavg[i][1]
        fid.write("{} {} {:>10.4f}\n".format(i+1,25.,aavg[i][0]))
        ftyp.write("Type {}: ({})--({})--({})\n".format(i+1,atype[0]+1,atype[1]+1,atype[2]+1))

    # Dihedral Coeffs
    fid.write("\nDihedral Coeffs\n\n")
    ftyp.write("\n~~~~~DIHEDRAL TYPES~~~~~\n")
    for i,dtype in enumerate(dtypes):
        fid.write("{} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}\n".format(i+1,0.0,0.0,0.0,0.0))
        ftyp.write("Type {}: ({})--({})--({})--({})\n".format(i+1,dtype[0]+1,dtype[1]+1,dtype[2]+1,dtype[3]+1))

    # CHECK FOR CHARGE NEUTRALIZATION
    qtot = 0.0
    ntot = 0.0
    for j,mol in enumerate(CGmols):
        for i in sorted(mol.keys()):
            bead = mol[i]
            ityp= bead['type']
            qtot  += CGmap[ityp]['charge']
            ntot  += 1.0
    print("# The total charge in the CG system is ", round(qtot,6))
    qavg = qtot/ntot
    print("# Subtracting residual of ", round(qavg,6), " to neutralize...")
    qtot = 0.0
    for j,mol in enumerate(CGmols):
        for i in sorted(mol.keys()):
            bead = mol[i]
            ityp= bead['type']
            qtot  += CGmap[ityp]['charge']
    print("# Now total system charge is ", round(qtot,6))

    # Atoms
    fid.write("\nAtoms\n\n")
    for j,mol in enumerate(CGmols):
        for i in sorted(mol.keys()):
            bead = mol[i]
            idi = bead['id']
            ityp= bead['type']
            ri  = bead['coords'].copy()
            #ri  = wrap_into_box(ri,box)
            qi  = CGmap[ityp]['charge'] - qavg
            fid.write("{} {} {} {:>8.3f} {:>15.5f} {:>15.5f} {:>15.5f} 0 0 0\n".format(idi,j+1,ityp+1,qi,\
                    bead['coords'][0],bead['coords'][1],bead['coords'][2]))

    # Bonds
    fid.write("\nBonds\n\n")
    for i,(bond,btype) in enumerate(zip(bonds,bond2typ)):
        fid.write("{} {} {} {}\n".format(i+1,btype+1,bond[0],bond[1]))

    # Angles
    fid.write("\nAngles\n\n")
    for i,(angle,atype) in enumerate(zip(angles,ang2typ)):
        fid.write("{} {} {} {} {}\n".format(i+1,atype+1,angle[0],angle[1],angle[2]))

    # Dihedrals
    fid.write("\nDihedrals\n\n")
    for i,(dihedral,dtype) in enumerate(zip(dihedrals,dih2typ)):
        fid.write("{} {} {} {} {} {}\n".format(i+1,dtype+1,dihedral[0],dihedral[1],dihedral[2],dihedral[3]))

    fid.close()
    ftyp.close()
    return

#==================================================================
# AUX: write_CG_lammpstrj
#==================================================================
def write_CG_lammpstrj(CGatoms,fid,timestep,box):
  N   = len(CGatoms)
  # write the header
  fid.write("ITEM: TIMESTEP\n\t{}\n".format(timestep))
  fid.write("ITEM: NUMBER OF ATOMS\n\t{}\n".format(N))
  fid.write("ITEM: BOX BOUNDS pp pp pp\n")
  fid.write("{:>10.4f}{:>10.4f}\n".format(box[0,0],box[0,1]))
  fid.write("{:>10.4f}{:>10.4f}\n".format(box[1,0],box[1,1]))
  fid.write("{:>10.4f}{:>10.4f}\n".format(box[2,0],box[2,1]))
  fid.write("ITEM: ATOMS id type x y z\n")

  # write the coordinates
  for i in sorted(CGatoms.keys()):
      CGatom = CGatoms[i]
      crds = CGatom['coords'][:].copy()
      crds = wrap_into_box(crds,box)
      typ  = CGatom['type']
      Id   = CGatom['id']
      fid.write("{:<10d}{:<4d}{:>12.4f}{:>12.4f}{:>12.4f}\n".format(Id,typ+1,crds[0],crds[1],crds[2]))
  return

#==================================================================
# AUX: write_CG_map
#==================================================================
def write_CG_map(CGatoms,fid):
  N   = len(CGatoms)
  mass = 0.0
  for i in sorted(CGatoms.keys()):
      fid.write("{:>6d} {:>5d} {:>10.5f} {:>8.3f} {:>6d}".format(\
              CGatoms[i]['id'],CGatoms[i]['type']+1,CGatoms[i]['mass'],CGatoms[i]['charge'],i))
      for j in sorted(CGatoms[i]['in']):
          fid.write(" {:>6d}".format(j))
      fid.write("\n")
  return

#==================================================================
# AUX: write_CG_pdb
#==================================================================
def write_CG_pdb(CGatoms,atoms,atmMap,fid):
  N   = len(CGatoms)
  fid.write("COMPND    UNNAMED\n")
  fid.write("AUTHOR    GENEATED BY CG_mapping.py\n")
  imap = {}
  cnt  = 0
  for i in sorted(CGatoms.keys()):
    cnt += 1
    imap[i] = cnt
    CGatom = CGatoms[i]
    ptr  = atmMap[i]
    crds = CGatom['coords'][:]
    typ  = atoms[ptr].el
    lbl  = atoms[ptr].name
    fid.write("{:<6s}{:>5d} {:<4s}{:1s}{:>3s}  {:>4d}{:<1s}   {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}      {:>4s}{:<s}\n".\
                            format("HETATM",imap[i],lbl[:3]," ","UNL",1," ",\
                                         crds[0],crds[1],crds[2],1.00,0.00,"",typ))
  for i in sorted(CGatoms.keys()):
    CGatom = CGatoms[i]
    ptr  = atmMap[i]
    crds = CGatom['coords'][:]
    typ  = atoms[ptr].el
    lbl  = atoms[ptr].name
    fid.write("{:<6s}{:>5d}".format("CONECT",imap[i]))
    for cnct in CGatom['adj']:
      fid.write("{:>5d}".format(imap[cnct]))
    fid.write("\n")
  fid.close()
  return None

#==================================================================
#  AUX: write_CG_xyz
#==================================================================
def write_CG_xyz(CGatoms,atoms,map,fid):
  N   = len(CGatoms)
  fid.write("{:<} \n\n".format(N))
  for i,CGatom in CGatoms.items():
    ptr  = atmMap[i]
    crds = CGatom['coords'][:]
    typ  = atoms['type'][ptr]
    lbl  = map[typ]
    fid.write('{:<10s} {:>15.3f} {:>15.3f} {:>15.3f}\n'.format(lbl,crds[0],crds[1],crds[2]))

#==================================================================
#  AUX: write_xyz
#==================================================================
def write_xyz(atoms,map,fid):
  N   = len(atoms['type'])
  fid.write("{:<} \n\n".format(N))
  for atm,ptr in atmMap.items():
    crds = atoms['coords'][ptr][:]
    typ  = atoms['type'][ptr]
    lbl  = map[typ]
    fid.write('{:<10s} {:>15.3f} {:>15.3f} {:>15.3f}\n'.format(lbl,crds[0],crds[1],crds[2]))

#==================================================================
#  AUX: assign_values
#==================================================================
def assign_values(atoms,mapFile,key):
  if mapFile is not None:
    print(mapFile)
    if mapFile != "pdb":
      theMap = process_map(mapFile)
      for atom in atoms:
        setattr(atom,key,theMap[atom.name])
  else:
    if key == 'mass':
      for atom in atoms:
        setattr(atom,key,guess_mass(atom.el))
    elif key == 'pval':
      for atom in atoms:
        pval = 1.0/round(atom.mass) if atom.mass > mlight else -1
        setattr(atom,key,pval)
    elif key == 'q':
      for atom in atoms:
        setattr(atom,key,0.0)


#==================================================================
#  AUX: process_map
#==================================================================
def process_map(mapFile):
  theMap = {}
  fid = open(mapFile,"r")
  lines = [line.strip().split() for line in fid]
  for line in lines:
    if is_number(line[1]):
      theMap[line[0]] = float(line[1])
    else:
      theMap[line[0]] = line[1]
  return theMap

#==================================================================
#  AUX: assign_mols
#==================================================================
# determines all molecules as disconnected subgraphs on a graph
def assign_mols(atoms,bonds):
  atmMap   = {atom.id : i for i,atom in enumerate(atoms)}
  untested = sorted(atmMap.keys())  # add everyone to being untested
  tested   = []                     # initialize list for tracking who has been tested
  queue    = []                     # initialize queue list
  mols     = []
  print("# Total number of atoms to be assigned:", len(untested))
  while untested:
    wait  = []                           # initialize wait list
    if not queue:                        # add to queue list if necessary
      queue.append(untested[0])
      mols.append([])
    for i in queue:                      # go through current queue list
      neighbors = bonds[i]               # find neighbor atoms
      mols[-1].append(i)                # add to current molecule
      neighbors = [ni for ni in neighbors if ni not in tested and ni not in queue] # only explore if untested/not in queue
      tested.append(i)                         # add i to tested listed
      untested.pop(untested.index(i))          # remove i from untested list
      wait.extend(neighbors)                   # add neighbors to wait list
    queue = list(set(wait[:]))

  print("# Total number of molecules is ", len(mols))
  the_mols = [0]*len(mols)
  for i,mol in enumerate(mols):
    print("# Number of atoms in mol ", i, ": ", len(mol))
    the_mols[i] = sorted(mol)
    for j in mol:
      atoms[atmMap[j]].molid = i

  return atmMap,the_mols

#==================================================================
#  AUX: get_bead_coords
#==================================================================
def get_bead_coords(chain,atoms):
  nbeads = len(chain['groups'])
  coords = np.zeros([nbeads,3])
  for i,g in enumerate(chain['groups']):   # for each group
    for j in g:				   # for each atom in group
      idj = atmMap[j]
      coords[i,:] += atoms['coords'][idj]*atoms['mass'][idj] # add mass-weighted coordinate
    coords[i,:] /= chain['mass'][i]
  return coords

#==================================================================
#  AUX: prioritize
#==================================================================
def prioritize(thisId,avail,atoms,adjlist,atmMap):
  # first screen by connectivity
  rank = [len(adjlist[i]) for i in avail]
  pval = [atoms[atmMap[i]].pval for i in avail]
  maxC = max(rank)
  # if there are equal connectivity proceed to priority numbers
  if (rank.count(maxC) > 1 ):
    rank = [pi if ri == maxC else 0 for ri,pi in zip(rank,pval)]
    maxC = max(rank)
    if (rank.count(maxC) > 1):
        printf("# Priority is ambiguous for assignment of atom {} to the following: ".format(thisId))
        for theId,theRank in zip(avail,rank):
            if theRank==maxC:
              printf("{} ".format(theId))
            printf("\n# Consider a different set of priority values. Just rolling with the first one for now...\n")
    return avail[rank.index(maxC)]
  else:
    return avail[rank.index(maxC)]

#==================================================================
#  AUX: get_CG_coords
#==================================================================
def get_CG_coords(mol,atoms,atmMap):
  for node,group in mol.items():
    nodeId = atmMap[node]
    mi     = atoms[nodeId].mass
    crds   = mi*atoms[nodeId].xs[:]
    M      = mi
    for i in group['in']:
      iId  = atmMap[i]
      mi   = atoms[iId].mass
      crds+= mi*atoms[iId].xs[:]
      M   += mi
    crds /= M
    group['coords'] = crds[:]

  return mol

#==================================================================
#  AUX: make_weight_groups
#==================================================================
def make_weight_groups(weights,neighbors):
  unique = list(np.unique(weights))
  groups = [[] for i in range(len(unique))]
  for i, wi in enumerate(weights):
    j = unique.index(wi)
    groups[j].extend([neighbors[i]])
  return groups

#==================================================================
#  AUX: add_if_heavy
#==================================================================
def add_if_heavy(node,neighbor,beads,atoms,atmMap):
  nid=atmMap[neighbor]
  mi = atoms[nid].mass
  if mi > mlight:
    beads[node]['nheavy'] +=1
  if neighbor in beads:               # if neighbor is existing bead
    for j in beads[neighbor]['in']:   # examine its constituents
      if j not in beads[node]['in']:  # if constituents not in present bead (MW: why would this happen???)
        mj = atoms[atmMap[j]].mass    # add if a heavy atom
        if mj > mlight:
          beads[node]['nheavy'] +=1
  return

#==================================================================
#  AUX: how_many_heavy
#==================================================================
def how_many_heavy(node,nlist,beads,atoms,options):
    nheavy = 0
    for neighbor in nlist:
        nid=atmMap[neighbor]
        mi = atoms[nid].mass
        if mi > mlight:
          nheavy += 1
        if neighbor in beads:
          for j in beads[neighbor]['in']:
              if j not in beads[node]['in']:
                  mj = atoms[atmMap[j]].mass
                  if mj > mlight:
                    nheavy +=1
    return nheavy

#==================================================================
#  AUX: get_mass
#==================================================================
def get_mass(atoms,node,group,atmMap):
    m = atoms[atmMap[node]].mass
    for i in group:
        m+= atoms[atmMap[i]].mass
    return m

#==================================================================
#  AUX: contract
#==================================================================
def contract(curr,touched,queue,major,minor,atoms,atmMap,options):

  # determine if major node will absorb the minor node
  mmajor     = get_mass(atoms,major,curr[major]['in'],atmMap)
  mminor     = get_mass(atoms,minor,curr[minor]['in'],atmMap)
  if mmajor + mminor > options.max_size:
      foundNode = False
  else:
      foundNode = True
      touched.add(major)
  if foundNode:
      curr[major]['in'].add(minor)
      add_if_heavy(major,minor,curr,atoms,atmMap)
      curr[minor]['adj'].remove(major)                      # remove contractor from adjacency list of contracted
      curr[major]['adj'].remove(minor)                      # remove contracted from adjacency list of contractor
      curr[major]['adj'].update(curr[minor]['adj'])         # resolve the adjacency list of contracted
      curr[major]['in'].update(curr[minor]['in'])           # resolve the container list of contracted
      # modify adjacency lists of neighbors to point to new node
      for i in curr[minor]['adj']:
        curr[i]['adj'].remove(minor)
        curr[i]['adj'].add(major)
      del curr[minor]                          # remove contracted from current list
      while (minor in queue):
          queue.pop(queue.index(minor))
      while (major in queue):
          queue.pop(queue.index(major))
  touched.add(minor)

  return foundNode,curr,touched,queue

def init_queue(beads,atoms,touched):
    queue = []
    queue = [i for i,beadi in beads.items()]
    istouched = [1 if i in touched else 0 for i in queue ]
    queue = [i for i in queue if i not in touched]
    return queue

#==================================================================
#  AUX: make_level_queue
#==================================================================
def make_level_queue(beads,lvl,atoms,touched):
  print("# Generating queue for coordination level ", lvl, "...")
  queue = []
  queue = [i for i,beadi in beads.items() if len(beadi['adj']) == lvl ]
  print("# There are currently ", len(queue), " groups with this level of connectivity...")
  istouched = [1 if i in touched else 0 for i in queue ]
  print("# Of these, ", sum(istouched), " have already been touched...")
  queue = [i for i in queue if i not in touched]
  return queue

#==================================================================
#  AUX: pageRank
#==================================================================
def pageRank(options,queue,nodes,touched,iIter):
    nodeList = sorted(nodes.keys())
    N = len(nodeList)
    A = np.zeros([N,N])
    Dsqrt = np.zeros([N,N])
    D = np.zeros([N,N])

    # GET TOTAL MASS
    if options.style == "mass" or options.style == "diff":
        mtot = 0.0
        ntot = 0.0
        mmax = 0.0
        mmin = 1e7
        for i,ni in enumerate(nodeList):
            mtot+= nodes[ni]['mass']
            mmax = max([mmax,nodes[ni]['mass']])
            mmin = min([mmin,nodes[ni]['mass']])
            ntot += 1
        mavg = mtot / ntot

    # construct standard adjacency matrix and the d
    for i,ni in enumerate(nodeList):
        jlinks    = [nodeList.index(link) for link in nodes[ni]['adj']]
        degree = len(jlinks)

        if options.style == "mass":
            A[i,i] =  nodes[ni]['mass']/mmax
        elif options.style == "diff":
            A[i,i] = (nodes[ni]['mass'] - mmin)/mavg
        if degree > 0:
          Dsqrt[i,i]= 1.0 #/ degree**0.5
          D[i,i]= degree  #/ degree**0.5
        for j in jlinks:
            A[i,j] = 1.0
            A[j,i] = 1.0
    print(A)
    # COMPUTE MAX EIGENVALUE
    alphaMax, Vmax = power_iteration(A)
    Vmax /= Vmax[0]
    sortVec = [x for x in Vmax]

    # this is the list sorted by eigenvector for largest eigenvalue
    sortList = [x for _, x in sorted(zip(sortVec,nodeList), key=lambda pair: (pair[0],pair[1]))]
    rank = [sortList.index(bead) for bead in queue]
    queue= [q for (r,q) in sorted(zip(rank,queue))]

    weights = {}
    fid = open("evec.dat.{}".format(iIter),"w")
    i = 1
    for weight,node in zip(sortVec,nodeList):
        weights[node] = weight
        fid.write("{} {} {}\n".format(node,float(weight),float(alphaMax)))
        i+=1
    fid.close()

    return queue,weights

#==================================================================
#  AUX: reorder_queue
#==================================================================
def reorder_queue(queue,touched,beads,tried):
  # ordering 1
  req  = [i for i in queue]
  return req

#==================================================================
#  AUX: get_overlap
#==================================================================
def get_overlap(listi,listj):
    n = 0
    ni= len(listi)
    nj= len(listj)
    for el in listi:
        if el in listj:
            listj.pop(listj.index(el))
            n += 1
    return float(n)/float(max(ni,nj))

#==================================================================
#  AUX: add_type
#==================================================================
def add_type(options,atoms,Id,group):
    typ    = atoms[Id].name
    if options.typing == "heavy":
        if atoms[Id].mass > mlight:
            group['type'].append(typ)
    else:
        group['type'].append(typ)
    return

#==================================================================
#  AUX: update_charge
#==================================================================
def update_charge(beadList,atoms,atmMap):
    for node,bead in beadList.items():
        iId = atmMap[node]
        q = atoms[iId].q
        for ib in bead['in']:
            jId = atmMap[ib]
            q += atoms[jId].q
        bead['charge'] = q
    return

#==================================================================
#  AUX: update_masses
#==================================================================
def update_masses(beadList,atoms,atmMap):
    for node,bead in beadList.items():
        iId = atmMap[node]
        m = atoms[iId].mass
        for ib in bead['in']:
            jId = atmMap[ib]
            m += atoms[jId].mass
        bead['mass'] = m
    return

#==================================================================
#  AUX: temp_types
#==================================================================
def temp_types(iRep,options,atoms,atmMap,beadsList):
    print("\n#=====================================")
    print("# Typing Summary for representation {}".format(iRep))
    print("#=====================================")
    # aggregate and count constituent atom types
    typeList = []

    # iterate over beads and also assign indices
    beadId = 0
    for beads in beadsList:
        for node in sorted(beads.keys()):
            group = beads[node]
            beadId +=1
            group['id'] = beadId
            nodeId = atmMap[node]
            add_type(options,atoms,nodeId,group)
            for j in group['in']:
                jId = atmMap[j]
                add_type(options,atoms,jId,group)
            theType= [[x,group['type'].count(x)] for x in set(group['type'])]   # count constituent atom types
            theType= [el for el in sorted(theType,key=lambda pair:pair[0])]     # organize in ascending order
            if theType not in typeList:
              typeList.append(theType)
            group['type'] = theType

    # sort the list of possible types for reduction
    nInType = [len(x) for x in typeList]
    typeList= [t for (t,n) in sorted(zip(typeList,nInType), key=lambda pair: pair[1])]

    # Assign all the atom types
    nOfType = [0 for t in typeList]
    for beads in beadsList:
        for node,group in beads.items():
            iType = typeList.index(group['type'])
            group['type']   = iType
            nOfType[iType] += 1

    # Check for similarity in constitution
    uniqueTypes = []
    uniqueExpnd = []
    nunique = 0
    queue   = typeList[:]
    typeMap = {}
    while queue:
        ti = queue.pop()                    # take something out of queue
        listi = []
        # generate the expanded list
        for el in ti:
            listi.extend([el[0]]*el[1])
        # check for similarity to existing unique types
        simScore    = [0]*nunique
        maxSimScore = -1.0
        for j in range(nunique):
            listj = uniqueExpnd[j][:]
            simScore[j] = get_overlap(listi,listj)
        if nunique > 0:
            maxSimScore = max(simScore)
            imax        = simScore.index(maxSimScore)
        if maxSimScore >= options.simrat:
            typeMap[typeList.index(ti)] = simScore.index(maxSimScore)
        else:
            uniqueTypes.append(ti[:])
            uniqueExpnd.append(listi[:])
            typeMap[typeList.index(ti)] = nunique
            nunique += 1

    # re-assign all the atom types
    nOfType = [0 for i in range(nunique)]
    for beads in beadsList:
        for node,group in beads.items():
            group['type']   = typeMap[group['type']]
            nOfType[group['type']] += 1

    # get average properties for the CG types
    CGmap = {}
    for i in range(nunique):
        CGmap[i] = {'mass': 0.0, 'charge': 0.0}
    for beads in beadsList:
        for i, group in beads.items():
            iCGtype= group['type']
            iId   = atmMap[i]
            iType = atoms[iId].name
            mi    = atoms[iId].mass
            qi    = atoms[iId].q
            CGmap[iCGtype]['mass']   += mi
            CGmap[iCGtype]['charge'] += qi
            for j in group['in']:
                jId = atmMap[j]
                jType = atoms[jId].name
                mj    = atoms[jId].mass
                qj    = atoms[jId].q
                CGmap[iCGtype]['mass']   += mj
                CGmap[iCGtype]['charge'] += qj

    for i,CGtype in CGmap.items():
        CGtype['mass']   /= nOfType[i]
        CGtype['charge'] /= nOfType[i]

    # write out summary
    print("# Total number of CG beads: {}".format(len(beads)))
    print("# Total number of CG types: {}".format(nunique))
    print("             {:^5s} {:^10s} {:^10s}".format("Ncg","<mass>","<charge>"))
    for i in range(nunique):
        print("-CG Type {:>3d}: {:^5d} {:>10.3f} {:>10.3f}".format(i+1,nOfType[i],CGmap[i]['mass'],CGmap[i]['charge']))

    return


#==================================================================
#  AUX: assign_CG_types
#==================================================================
def assign_CG_types(files,options,atoms,atmMap,beadsList):
    # aggregate and count constituent atom types
    typeList = []

    # iterate over beads and also assign indices
    beadId = 0
    for beads in beadsList:
      for node in sorted(beads.keys()):
        group = beads[node]
        beadId +=1
        group['id'] = beadId
        nodeId = atmMap[node]
        add_type(options,atoms,nodeId,group)
        for j in group['in']:
          jId = atmMap[j]
          add_type(options,atoms,jId,group)
        theType= [[x,group['type'].count(x)] for x in set(group['type'])]   # count constituent atom types
        theType= [el for el in sorted(theType,key=lambda pair:pair[0])]     # organize in ascending order
        if theType not in typeList:
          typeList.append(theType)
        group['type'] = theType

    # sort the list of possible types for reduction
    nInType = [len(x) for x in typeList]
    typeList= [t for (t,n) in sorted(zip(typeList,nInType), key=lambda pair: pair[1])]

    # Assign all the atom types
    nOfType = [0 for t in typeList]
    for beads in beadsList:
      for node,group in beads.items():
        iType = typeList.index(group['type'])
        group['type']   = iType
        nOfType[iType] += 1

    # Check for similarity in constitution
    uniqueTypes = []
    uniqueExpnd = []
    nunique = 0
    queue   = typeList[:]
    typeMap = {}
    while queue:
      ti = queue.pop()                    # take something out of queue
      listi = []
      # generate the expanded list
      for el in ti:
        listi.extend([el[0]]*el[1])
      # check for similarity to existing unique types
      simScore    = [0]*nunique
      maxSimScore = -1.0
      for j in range(nunique):
        listj = uniqueExpnd[j][:]
        simScore[j] = get_overlap(listi,listj)
      if nunique > 0:
        maxSimScore = max(simScore)
        imax        = simScore.index(maxSimScore)
      if maxSimScore >= options.simrat:
        typeMap[typeList.index(ti)] = simScore.index(maxSimScore)
      else:
        uniqueTypes.append(ti[:])
        uniqueExpnd.append(listi[:])
        typeMap[typeList.index(ti)] = nunique
        nunique += 1

    # re-assign all the atom types
    nOfType = [0 for i in range(nunique)]
    for beads in beadsList:
      for node,group in beads.items():
        group['type']   = typeMap[group['type']]
        nOfType[group['type']] += 1

    # get average properties for the CG types
    CGmap = {}
    for i in range(nunique):
      CGmap[i] = {'mass': 0.0, 'charge': 0.0}
    for beads in beadsList:
      for i, group in beads.items():
        iCGtype= group['type']
        iId   = atmMap[i]
        iType = atoms[iId].name
        mi    = atoms[iId].mass
        qi    = atoms[iId].q
        CGmap[iCGtype]['mass']   += mi
        CGmap[iCGtype]['charge'] += qi
        for j in group['in']:
          jId = atmMap[j]
          jType = atoms[jId].name
          mj    = atoms[jId].mass
          qj    = atoms[jId].q
          CGmap[iCGtype]['mass']   += mj
          CGmap[iCGtype]['charge'] += qj

    for i,CGtype in CGmap.items():
      CGtype['mass']   /= nOfType[i]
      CGtype['charge'] /= nOfType[i]

    # write out summary
    fid = open("typing.summary.txt","w")
    fid.write("#===========================\n")
    fid.write("# Typing Summary\n")
    fid.write("#===========================\n")
    fid.write("# Total number of CG beads: {}\n".format(len(beads)))
    fid.write("# Total number of CG types: {}\n".format(nunique))
    fid.write("             {:^5s} {:^10s} {:^10s}\n".format("Ncg","<mass>","<charge>"))
    for i in range(nunique):
      fid.write("-CG Type {:>3d}: {:^5d} {:>10.3f} {:>10.3f}\n".format(i+1,nOfType[i],CGmap[i]['mass'],CGmap[i]['charge']))
    fid.write("\n~~~~~CG TYPES~~~~~\n")

    for i in range(nunique):
      fid.write("Type {}: ".format(i+1))
      for j in uniqueExpnd[i]:
          fid.write("{} ".format(j))
      fid.write("\n")

    return fid,CGmap,nOfType

#==================================================================
#  AUX: write_groups
#==================================================================
def write_groups(i,beads,atoms,map):
  fid = open("mol_" + repr(i) + ".groups.dat","w")
  for node,group in beads.items():
      nhvy = group['nheavy']
      nall = len(group['in']) + 1
      ptr  = atmMap[node]
      typ  = atoms[ptr].name
      lbl  = map[typ]
      fid.write("{}({}) {} {}-- ".format(node,lbl,nall,nhvy))
      for neighbor in group['adj']:
          ptr = atmMap[neighbor]
          typ = atoms[ptr].name
          lbl = map[typ]
          fid.write("{}({}) ".format(neighbor,lbl))
      fid.write("\n\t >> {} : ".format(len(group['in'])))
      for atom in group['in']:
          ptr = atmMap[atom]
          typ = atoms[ptr].name
          lbl = map[typ]
          fid.write("{}({}) ".format(atom,lbl))
      fid.write("\n")
  fid.close()
  return

#==================================================================
#  AUX: redution_mapping
#==================================================================
# determines the grouping of atoms that form CG beads. Information is in
# a dictionary with the following fields
# 'adj' - the adjacency list (set of atoms for bonding)
# 'in'  - constituent atoms
# 'nheavy' - number of heavy atoms in the CG atom
# 'type' - type of the CG bead
# 'coords' - coordinates of the bead
# 'id' - assigned ID
def reduction_mapping(files,options,moli,atoms,adjlist,atmMap):
  history = []
  curr = {}
  queue= []

  # Set up initial list and mark atoms to be reduced
  files['summary'].write("Initial Number of atoms: {}\n".format(len(moli)))
  print("# Beginning reduction mapping of molecule with {} atoms...".format(len(moli)))
  for i in moli:
    idi = atmMap[i] # key to index in rest of atoms structure
    if (atoms[idi].pval == -1):
      queue.append(i)
    else:
      mi = atoms[idi].mass
      qi = atoms[idi].q
      if mi > mlight:
        curr[i] = {'adj' : set(adjlist[i]),'in': set(), 'nheavy':1, 'type': [],'mass': mi,'charge':  qi}
      else:
        curr[i] = {'adj' : set(adjlist[i]),'in': set(), 'nheavy':0, 'type': [],'mass': mi,'charge': qi}
  print("# {} atoms marked to be contracted into remaining {} groups".format(len(queue),len(curr)))

  # Perform initial contraction for negative priority atoms
  files['summary'].write("Initial contraction consists of {} into {} groups\n".format(len(queue),len(curr)))
  for i in queue:
    neighbors = list(adjlist[i])                           # determine the set of available neighbors
    mergeId   = prioritize(i,neighbors,atoms,adjlist,atmMap)   # find who to contract into
    neighbors.pop(neighbors.index(mergeId))   # remove contractor from the available neighbors of contracted
    curr[mergeId]['in'].add(i)                # augment list to reflect the contraction
    add_if_heavy(mergeId,i,curr,atoms,atmMap)
    curr[mergeId]['adj'].remove(i)            # remove the contracted from adjacency list of contractor
    curr[mergeId]['adj'].update(neighbors)    # resolve the adjacency list of contracted
  update_masses(curr,atoms,atmMap)
  update_charge(curr,atoms,atmMap)
  history.append(copy.deepcopy(curr))

  # Start coordination level reductions, contracting from degree 2 and up
  for iIter in range(options.niter):
    print_status("Iteration {}".format(iIter+1))
    if len(curr) > 1:
      files['summary'].write("Reduction Round {}\n".format(iIter+1))
      files['summary'].write("Initial number of groups: {}\n".format(len(curr)))
      touched = set()
      queue         = init_queue(curr,atoms,touched)
      queue,weights = pageRank(options,queue,curr,touched,iIter) # gives page rank queue
      wlast = weights[queue[0]]
      tlist = set()
      print("# There are {} nodes in the queue...".format(len(queue)))
      while queue:
        node = queue.pop(0)     # obtain index for first in queue
        wcurr = weights[node]
        if wcurr != wlast:
          for major in tlist:
            touched.add(major)
          tlist = set()
        wlast = wcurr

        # check mass to see if available for contraction
        if get_mass(atoms,node,curr[node]['in'],atmMap) >= options.max_size:
          touched.add(node)
        else:
          neighbors = list(curr[node]['adj'])   # set of neighbors
          neighbors = [n for n in neighbors if n not in touched]
          wneigh    = np.array([weights[j] for j in neighbors])
          neighbors = [n for w,n in sorted(zip(wneigh,neighbors),\
                  key=lambda pair: (pair[0],pair[1])) if w >= weights[node]] # sorted by weights
          wneigh = [weights[j] for j in neighbors]
          ngroups = make_weight_groups(wneigh,neighbors)
          for majorgroup in ngroups:
            tryNextGroup = True
            while majorgroup:
              major = majorgroup.pop(0)
              foundNode,curr,tlist,queue = contract(curr,tlist,queue,major,node,atoms,atmMap,options)
              if foundNode:
                  tryNextGroup = False
                  node = major
            if not tryNextGroup:
              break

      update_masses(curr,atoms,atmMap)
      update_charge(curr,atoms,atmMap)
      print("# Queue is exhausted and vertex groups formed...")
      print("# There are currently {} vertex groups...".format(len(curr)))
    else:
      print("There is only one group left! No more coarsening possible")
    files['summary'].write("Reduction at iteration {} --> {} groups\n".format(iIter,len(curr)))
    history.append(copy.deepcopy(curr))

  files['summary'].write("\n\n")
  return curr,history

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#  SUB-MAIN: CGmapping
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def CGmapping(files,options):
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # SCREEN FILES AND GENERATE MAPS
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  print_status("Screening PDB file and assigning attributes")
  atoms      = pdb.process_atoms(files['pdb'])
  assign_values(atoms,files['mmap'],'mass')
  assign_values(atoms,files['pmap'],'pval')
  assign_values(atoms,files['qmap'],'q')
  bonds = pdb.process_bonds(files['pdb'])
  atmMap,mols = assign_mols(atoms,bonds)
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  # PERFORM THE MAPPING FOR EACH MOLECULE
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  print_status("Performing mapping")
  CGmol   = []
  CGhist  = []
  fpdb,fmap,fout = open_files(options,mols)
  for i,moli in enumerate(mols):
    print_status("Mapping of molecule {}".format(i))
    files['summary'].write("Reduction Summary for molecule {}\n\n".format(i))
    CGmoli,histi = reduction_mapping(files,options,moli,atoms,copy.deepcopy(bonds),atmMap)
    CGmol.append(CGmoli)
    CGhist.append(histi)

  # ASSIGN CG ATOM TYPES
  print_status("Assigning preliminary CG site types")
  ftyp,CGtypeMap,nCgType = assign_CG_types(files,options,atoms,atmMap,CGmol)

  # GET TYPES AT EACH HISTORY LEVEL
  nhist = len(CGhist[0])
  nmol  = len(CGhist)
  tmpCGmol  = [[] for i in range(nhist)]
  for i in range(nhist):
    for j in range(nmol):
      tmpCGmol[i].extend([CGhist[j][i]])
  for i in range(nhist):
    temp_types(i,options,atoms,atmMap,tmpCGmol[i])

  # WRITE OUT COORDINATES FOR THE GROUPS
  print_status("Writing CG Coordinate files...")
  molcpy = []
  for i,CGmoli in enumerate(CGmol):
    for j,histj in enumerate(CGhist[i]):
      tmp = get_CG_coords(copy.deepcopy(histj),atoms,atmMap)
      write_CG_pdb(tmp,atoms,atmMap,fpdb[i][j])
      write_CG_map(tmp,fmap[j])

  print_status("Wrapping up...")

  return True

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#  MAIN: _main
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
def main(argv):
  # PARSE ARGUMENTS
  parser = create_parser()
  args                         = parser.parse_args()
  (files,options)              = convert_args(args)

  # BEGIN MAPPING
  t0 = time.time()
  CGmapping(files,options)
  print("# The elapsed time is {} seconds".format(time.time()-t0))

  return True

#==================================================================
#  RUN PROGRAM
#==================================================================
if __name__ == "__main__":
  main(sys.argv[1:])
