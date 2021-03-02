import numpy as np
import matplotlib.pyplot as plt

def drude(energy, 
	eps_inf, 
	wp, 
	gam, 
	):
	drude = eps_inf - wp**2/(energy**2+1j*gam*energy)
	eps_real = np.real(drude)
	eps_imag = np.imag(drude)
	n = np.sqrt( (np.sqrt(eps_real**2 + eps_imag**2) + eps_real)/2.0 )
	k = np.sqrt( (np.sqrt(eps_real**2 + eps_imag**2) - eps_real)/2.0 )
	return n, k

def write_func(energy, eps_inf, wp, gam):
	n, k = drude(energy=energy, eps_inf=eps_inf, wp=wp, gam=gam)
	title = str('% Gold Dielectric (wp=')+str(wp)+str(', eps_inf=')+str(eps_inf)+str(', gam=')+str(gam)+ str(')')+'\n'
	file = open(str('au_drude.dat'),'w')
	file.write(title)
	file.write( str('% Energy [eV] 	n 	k') + '\n')
	for j in range(0, len(energy)):
		file.write( "%.3f" % energy[j] + '\t' + "%.5f" % n[j] + '\t' + "%.5f" % k[j] + '\n')
	file.close()

energy_range = np.round(np.linspace(.535,4,100),3)


write_func(energy=energy_range, eps_inf=9.695, wp=8.959, gam=0.073)

