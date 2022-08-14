import sys
sys.path.append("../GO")

from Bio import pairwise2

alignments = pairwise2.align.localxx("MNANSTTTAI", "NANS")
print(pairwise2.format_alignment(*alignments[0]))
