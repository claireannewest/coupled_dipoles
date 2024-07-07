import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import derivative
from scipy.special import spherical_jn
from scipy.special import spherical_yn
    
e = 4.80326E-10 # elementary charge, [statC, g^1/2 cm^3/2 s^-1]
c = 2.998E+10 # speed of light [cm/s]
hbar_eVs = 6.58212E-16 # Planck's constant [eV*s]

class Sphere_Polarizability:
    def __init__(self, centers, all_radii, n, wp, eps_inf, gam_drude, shell=False, shell_m=False):
        """Defines the different system parameters.
        
        Keyword arguments:
        centers -- coordinates of centers of each particle [cm]
        all_radii -- radii of the oscillators [cm]
        n -- [unitless] refractive index of background 
        wp -- [1/s] bulk plasma frequency
        eps_inf -- [unitless] static dielectric response of ionic background 
        gamNR_qs -- [1/s] Drude damping (i.e. quasistatic, nonradiative)
        """
        self.centers = centers
        self.all_radii = all_radii
        self.n = n
        self.wp = wp
        self.eps_inf = eps_inf
        self.gamNR_qs = gam_drude
        self.shell = shell
        self.shell_m = shell_m

    def psi(self, rho):
        return rho*spherical_jn(1, rho)

    def psi_prime(self, rho):
        return spherical_jn(1, rho) + rho*spherical_jn(1, rho, derivative=True)

    def hankel(self, rho):
        return spherical_jn(1, rho) + 1j*spherical_yn(1, rho)

    def hankel_prime(self, rho):
        return spherical_jn(1, rho, derivative=True) + 1j*spherical_yn(1, rho,derivative=True)

    def xi(self, rho):
        return rho*self.hankel(rho)

    def xi_prime(self, rho):
        return self.hankel(rho) + rho*self.hankel_prime(rho)

    def mie_coefficents(self, w, radius):
        eps = self.eps_inf - self.wp**2/(w**2 + 1j*w*self.gamNR_qs)
        m = np.sqrt(eps)
        k = w/c*self.n
        x = k*radius
        numer_a = m*self.psi(m*x)*self.psi_prime(x) - self.psi(x)*self.psi_prime(m*x)
        denom_a = m*self.psi(m*x)*self.xi_prime(x) - self.xi(x)*self.psi_prime(m*x)
        numer_b = self.psi(m*x)*self.psi_prime(x) - m*self.psi(x)*self.psi_prime(m*x)
        denom_b = self.psi(m*x)*self.xi_prime(x) - m*self.xi(x)*self.psi_prime(m*x)
        a = numer_a/denom_a
        b = numer_b/denom_b
        return a 

    def chi(self,z):
        return -z*spherical_yn(1, z)

    def chi_prime(self,z):
        return -spherical_yn(1, z) + -z*spherical_yn(1, z, derivative=True)


    def mie_coeff_coated_sphere(self, w, radius):
        eps1 = self.eps_inf - self.wp**2/(w**2 + 1j*w*self.gamNR_qs)
        m1 = np.sqrt(eps1)
        m2 = self.shell_m
        k = w/c*self.n
        x = k*radius
        y = k*(radius+self.shell)
        A = ( m2*self.psi(m2*x)*self.psi_prime(m1*x) - m1*self.psi_prime(m2*x)*self.psi(m1*x) ) /\
            ( m2*self.chi(m2*x)*self.psi_prime(m1*x) - m1*self.chi_prime(m2*x)*self.psi(m1*x) )
        a = ( self.psi(y)*( self.psi_prime(m2*y) - A*self.chi_prime(m2*y) ) -  m2*self.psi_prime(y)*( self.psi(m2*y) - A*self.chi(m2*y) ) ) / \
            ( self.xi(y)* ( self.psi_prime(m2*y) - A*self.chi_prime(m2*y) ) -  m2*self.xi_prime(y)* ( self.psi(m2*y) - A*self.chi(m2*y) ) )
        return a 


    def alpha(self, w, radius, imageornaw):
        k = w/c*self.n
        if self.shell == False:
            a = self.mie_coefficents(w=w,radius=radius)
        if self.shell != False: 
            a = self.mie_coeff_coated_sphere(w=w, radius=radius)
        alpha = 3/(2.*k**3)*1j*(a)#+b) 
        if imageornaw is True:
            alpha = alpha * (-1)
        return alpha

class CrossSections:
    def __init__(self, 
        centers, 
        all_radii,
        w, 
        num,
        imageornaw,
        n, 
        wp, 
        eps_inf, 
        gam_drude,
        shell=False,
        shell_m=False,
        ):
        """Defines the different system parameters.
        
        Keyword arguments:
        centers -- coordinates of centers of each particle [cm]
        all_radii -- radii of the oscillators [cm]
        w -- array of frequencies [1/s]
        num -- number of particles 
        n -- [unitless] refractive index of background 
        wp -- [1/s] bulk plasma frequency
        gamNR_qs -- [1/s] Drude damping (i.e. quasistatic, nonradiative)
        """
        self.centers = centers
        self.all_radii = all_radii
        self.w = w
        self.num = num
        self.imageornaw = imageornaw
        self.n = n
        self.wp = wp
        self.eps_inf = eps_inf
        self.gamNR_qs = gam_drude
        self.shell = shell
        self.shell_m = shell_m
        self.dip_params = Sphere_Polarizability(self.centers, self.all_radii, self.n, self.wp, self.eps_inf, self.gamNR_qs, self.shell, self.shell_m)
        self.num_radii_combs = self.centers.shape[0]
        self.num_gap = self.centers.shape[1]
        self.num_freq = self.centers.shape[2]

    def delta_mn(self, m, n):
        if m == n:
            return 1
        else:
            return 0

    def A_ij(self, dip_i, dip_j):
        """ Off diagonal block terms in A_tilde. 
        """
        k = self.w/c*self.n
        A_ij = np.zeros( (self.num_radii_combs, self.num_gap, self.num_freq, 3, 3) ,dtype=complex) 
        r = self.centers[:,:,:,dip_i,:] - self.centers[:,:,:,dip_j,:] # [cm], distance between ith and jth particle 
        magr = np.sqrt(r[:,:,:,0]**2 + r[:,:,:,1]**2 + r[:,:,:,2]**2)
        for n in range(0,3):
            for m in range(0,3):
                A_ij[:,:,:,n,m] = np.exp(1j*k*magr) / magr**3 *(
                           k**2*( r[:,:,:,n]*r[:,:,:,m] - magr**2*self.delta_mn(n+1,m+1) ) + 
                            (1-1j*k*magr) / magr**2 *(magr**2*self.delta_mn(n+1,m+1)-3*r[:,:,:,n]*r[:,:,:,m]))
        return A_ij

    def alpha_tensor(self, dip_i):
        """ Polarizability tensor in the cartesian basis of the single particle.
            This assumes the sphere has an isotropic polarizability. 
        """
        alpha = np.zeros( (self.num_radii_combs, self.num_gap, self.num_freq, 3, 3) ,dtype=complex)
        alpha[:,:,:,0,0] = self.dip_params.alpha(w=self.w, radius=self.all_radii[:,:,:,dip_i], imageornaw=self.imageornaw[dip_i])
        alpha[:,:,:,1,1] = self.dip_params.alpha(w=self.w, radius=self.all_radii[:,:,:,dip_i], imageornaw=self.imageornaw[dip_i])
        alpha[:,:,:,2,2] = self.dip_params.alpha(w=self.w, radius=self.all_radii[:,:,:,dip_i], imageornaw=self.imageornaw[dip_i])
        return alpha

    def A_ii(self, dip_i, dip_j):
        """On diagonal block terms in A_tilde.
        """
        alpha = self.alpha_tensor(dip_i=dip_i)
        A_ii = np.linalg.inv(alpha)
        return A_ii

    def A_tilde(self):
        """A_tilde = [num_freq, 3*N, 3*N]
        """
        A_tilde = np.zeros( (self.num_radii_combs, self.num_gap, self.num_freq, 3*self.num, 3*self.num) ,dtype=complex) 
        for i in range(0 , self.num): 
            for j in range(0, self.num):
                if i == j:  
                    A_tilde[:, :, :, 3*i : 3*(i+1), 3*i : 3*(i+1)] = self.A_ii(dip_i=i, dip_j=i)
                if i != j:
                    A_tilde[:, :, :, 3*i : 3*(i+1), 3*j : 3*(j+1)] = self.A_ij(dip_i=i, dip_j=j)
        return A_tilde

    def P_tilde(self, drive):
        E0_tilde = np.zeros((self.num_radii_combs, self.num_gap, self.num_freq, 3*self.num, 1 ) )
        P_tilde = np.zeros(( self.num_radii_combs, self.num_gap, self.num_freq, 3*self.num, 1 ),dtype=complex)
        for i in range(0, self.num):
            E0_tilde[:, :, :, 3*i,:] = drive[0]
            E0_tilde[:, :, :, 3*i+1,:] = drive[1]
            E0_tilde[:, :, :, 3*i+2,:] = drive[2]
        P_tilde = np.linalg.solve(self.A_tilde(), E0_tilde)
        return P_tilde

    def cross_sects(self, drive):
        ''' Works for spheres up to 50 nm radii, and in the window < 3.0 eV '''
        k = self.w/c*self.n
        P = self.P_tilde(drive=drive)
        P_each = np.zeros((self.num_radii_combs, self.num_gap, self.num_freq, 3, self.num),dtype=complex)
        E_each = np.zeros((self.num_radii_combs, self.num_gap, self.num_freq, 3, self.num),dtype=complex)
        Cext_each = np.zeros((self.num_radii_combs, self.num_gap, self.num_freq, self.num))
        Cabs_each = np.zeros((self.num_radii_combs, self.num_gap, self.num_freq, self.num))
        for i in range(0, self.num):
            # Evaluate the cross sections of each particle separately
            P_each[:, :, :, :,i] = P[:, :, :, 3*i:3*(i+1), 0]
            E_each[:, :, :, 0,i] = self.dip_params.alpha(w=self.w, radius=self.all_radii[:, :, :, i],imageornaw=self.imageornaw[i])**(-1)*P_each[:, :, :, 0,i]
            E_each[:, :, :, 1,i] = self.dip_params.alpha(w=self.w, radius=self.all_radii[:, :, :, i], imageornaw=self.imageornaw[i])**(-1)*P_each[:, :, :, 1,i]
            E_each[:, :, :, 2,i] = self.dip_params.alpha(w=self.w, radius=self.all_radii[:, :, :, i], imageornaw=self.imageornaw[i])**(-1)*P_each[:, :, :, 2,i]

            Cext_each[:, :, :, i] = 4*np.pi*k*np.imag( np.sum( P_each[:, :, :,:, i]*np.conj(E_each[:,:, :,:, i]),axis=3 )) *10**8
            Cabs_each[:, :, :, i] = Cext_each[:, :, :, i] - 4*np.pi*k*2/3*k**3*np.real( np.sum(P_each[:, :, :,:, i]*np.conj(P_each[:,:, :,:, i]), axis=3) ) *10**8  

        return Cext_each, Cabs_each


