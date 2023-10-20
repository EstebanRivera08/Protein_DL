from numpy import identity
NUC = ["A" , "C" , "T" , "E" , "D" , "F" , "W" , "I" , "V" , "L" , "K" , "M" , "N" , "Q" , "S" , "R" , "Y" , "H" , "P" , "G", "-"]
NUC_P = [(a, b) for a in NUC for b in NUC]
NUC_B = identity(len(NUC), dtype=float).tolist()

NNUC_B = identity(len(NUC), dtype=float)
NNUC_B[NNUC_B == 0] = 1.
NNUC_B[-1] *= 0.

NNUC_B = NNUC_B.tolist()
