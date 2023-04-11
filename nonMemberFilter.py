from astropy.table import Table
import numpy as np
nameList = ["RA","Dec","(B-V)_0","M_u","M_B","M_V","u","B","V","I"]
dat = Table.read("nonMembers/M79_nonMembers.dat", format="ascii.no_header", delimiter="\t", fast_reader=False)

mv = dat['col7']
b = dat['col9']
v = dat['col10']

cond = np.logical_and.reduce((mv > 2.5, mv < 3.5, (b-v) > 0.23, (b-v) < 0.27))
dat1 = dat[cond]

infiles = np.savetxt("nonMembers/%s_nonMembers123.dat"%("M79"),dat1, fmt="%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.4f\t%.4f\t%.4f\t%.4f", delimiter = "\t",header="{} E(B-V)={:.2f}, m-M = {:.2f}\n\tRA\t\tDec\t\t(B-V)_0\tM_u\tM_B\tM_V\tu\tB\tV\tI".format("M79",0.01, 15.55))