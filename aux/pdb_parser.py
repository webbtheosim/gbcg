#!/usr/bin/env python
__author__ = "Michael A. Webb"
__version__ = "1.0.1"
__credits__ = ["Michael A. Webb"]
__email__ = "mawebb@princeton.edu"
__license__ = "GPL"

"""
Module containing a set of functions that are useful for parsing pdb files
"""
#==================================================================
# MODULES
#==================================================================
import numpy as np

#==================================================================
#  AUX: add_to_dict
#==================================================================
def add_to_dict(dic,key,val):
  if key in dic:
    dic[key].add(val)
  else:
    dic[key] = set([val])

#==================================================================
#  CLASS: ATOM
#==================================================================
class ATOM(object):
  def __init__(self,record):
    self.id   = int(record[1])
    self.molid= -1
    shift     = 1 if isfloat(record[-1]) else 0
    self.name = record[2] if shift else record[-1]
    self.el   = record[2]
    self.q    = float(record[-2+shift])
    self.pval = float(record[-3+shift])
    self.mass = 100.0
    self.xs   = np.array([float(x) for x in\
            record[-6+shift:-3+shift]])
    self.bonds= []

#==================================================================
#  AUX: isfloat
#==================================================================
def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

#==================================================================
#  AUX: get_adjacency_list
#==================================================================
def get_adj_list(datafile,atoms):

  # EXAMINE TOPOLOGY FROM DATA FILE
  adjlist = {};
  if datafile != None:
    print("# Extracting topology from {}.".format(datafile))
    fid = open(datafile)
    line = fid.readline().strip().split()
    while True:
      if (len(line) == 2 and line[1] == "bonds"):
        nbond=int(line[0])
        print("# A total of {} bonds reports!".format(nbond))
      if (len(line) == 1 and line[0] == "Bonds"):
        fid.readline()
        for j in range(nbond):
          line = fid.readline().strip().split()
          bond = [int(el) for el in line]
          if bond[2] in atoms['id'].keys():
            add_to_dict(adjlist,bond[2],bond[3])
            add_to_dict(adjlist,bond[3],bond[2])
        print("# Bonds field found and record! Breaking from file...")
        fid.close()
        break
      line = fid.readline().strip().split()
  else:
    atms = sorted(atoms['id'].keys())
    bonds = [[atms[i],atms[i+1]] for i in range(len(atms)-1)]
    for j, bond in enumerate(bonds):
      if bond[1] in atoms['id'].keys():
        add_to_dict(adjlist,bond[0],bond[1])
        add_to_dict(adjlist,bond[1],bond[0])
    print("# Assuming linear connectivity amongst particles based on indices")
  return adjlist

#==================================================================
#  AUX: process_bonds
#==================================================================
def process_bonds(pdb_file):

  # screen the pdb file to find all lines with 'CONECT'
  bondLines = []
  with open(pdb_file) as fid:
    line = fid.readline().strip()
    while line:
      if line[:6] == 'CONECT':
        tmpStr = line[6:]
        ids    = [tmpStr[i:i+5] for i in range(0,len(tmpStr),5)]
        splitLine = ['CONECT'] + ids
        bondLines.append(splitLine)
      line = fid.readline().strip()

    #line = fid.readline().strip().split()
    #while line:
    #  if line[0] == 'CONECT':
    #    bondLines.append(line)
    #  line = fid.readline().strip().split()
  print("# {} CONECT fields found...".format(len(bondLines)))

  # now add all the bonds to a dictionary
  # in pdb files there should be CONECT fields
  # for all of the atoms listed, and it is unneccessary
  # to perform the reverse addition, in principle
  # but we are doing it for safety
  bonds = {}
  for bondLine in bondLines:
    iAtm   = int(bondLine[1])
    bonded = [int(jAtm) for jAtm in bondLine[2:]]
    for jAtm in bonded:
      add_to_dict(bonds,iAtm,jAtm)
      add_to_dict(bonds,jAtm,iAtm)

  return bonds

#==================================================================
#  AUX: process_atoms
#==================================================================
def process_atoms(pdb_file):

  # screen the pdb file to find all lines with 'HETATM' or 'ATOM' records
  atomLines = []
  with open(pdb_file) as fid:
    line = fid.readline().strip().split()
    while line:
      if line[0] in ('HETATM','ATOM'):
        atomLines.append(line)
      line = fid.readline().strip().split()
  nAtm = len(atomLines)
  print("# {} atoms were found in {}".format(nAtm,pdb_file))

  # Now collect the atoms data
  atoms  = [-1]*nAtm # container for all atom data
  for atomLine in atomLines:
    theAtom = ATOM(atomLine)
    atoms[theAtom.id-1] = theAtom
  return atoms
