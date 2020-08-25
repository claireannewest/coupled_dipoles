import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.misc import derivative
from scipy.special import spherical_jn
from scipy.special import spherical_yn
    
e = 4.80326E-10 # elementary charge, [statC, g^1/2 cm^3/2 s^-1]
c = 2.998E+10 # speed of light [cm/s]
hbar_eVs = 6.58212E-16 # Planck's constant [eV*s]

class DipoleParameters:
    def __init__(self, centers, orient, all_radii, n, wp, eps_inf, gam_drude):
        """Defines the different system parameters.
        
        Keyword arguments:
        centers -- coordinates of centers of each prolate spheorid [cm]
        orient -- orientation (angle w.r.t. z axis) of the long axis of prolate spheroid 
        all_radii -- radii of the oscillators [cm]
        n -- [unitless] refractive index of background 
        wp -- [1/s] bulk plasma frequency
        eps_inf -- [unitless] static dielectric response of ionic background 
        gamNR_qs -- [1/s] Drude damping (i.e. quasistatic, nonradiative)
        """
        self.centers = centers
        self.orient = orient
        self.all_radii = all_radii
        self.n = n
        self.wp = wp
        self.eps_inf = eps_inf
        self.gamNR_qs = gam_drude

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

    def mie_coefficents(self, w,radius):
        eps = self.eps_inf - self.wp**2/(w**2 + 1j*w*self.gamNR_qs)
        m = np.sqrt(eps)
        k = w/c
        x = k*radius
        numer_a = m*self.psi(m*x)*self.psi_prime(x) - self.psi(x)*self.psi_prime(m*x)
        denom_a = m*self.psi(m*x)*self.xi_prime(x) - self.xi(x)*self.psi_prime(m*x)
        numer_b = self.psi(m*x)*self.psi_prime(x) - m*self.psi(x)*self.psi_prime(m*x)
        denom_b = self.psi(m*x)*self.xi_prime(x) - m*self.xi(x)*self.psi_prime(m*x)
        a = numer_a/denom_a
        b = numer_b/denom_b
        return a, b

    def alpha(self, w, radius, kind): 
        ### assumes z axis is long axis, and y = x 
        k = w/c
        # cs equals long axis, a equals short axis
        cs = radius[0]; a = radius[1]
        if cs == a:
            ### it's a sphere
            a, b = self.mie_coefficents(w,radius[0])
            alpha = 3/(2.*k**3)*1j*(a)#+b) 
        else:
            ### it's a spheroid 
            V = 4/3*np.pi*cs*a**2
            eps = self.eps_inf - self.wp**2/(w**2 + 1j*w*self.gamNR_qs)
            es = np.sqrt(1 - a**2/cs**2)
            Lz = (1-es**2)/es**2*(-1 + 1/(2*es)*np.log((1+es)/(1-es)))
            Dz = 3/4*((1+es**2)/(1-es**2)*Lz + 1) # [unitless]
            Ly = (1 - Lz)/2
            Dy = a/(2*cs)*( 3/es*np.arctanh(es) - Dz) 
            if kind == 'long':
                lE = cs
                L = Lz
                D = Dz
            if kind == 'short':
                lE = a
                L = Ly
                D = Dy
            alphaR = V/(4*np.pi)*(eps-1)/(1+L*(eps-1))
            alphaMW = alphaR / (1 - k**2/lE *D*alphaR - 1j*2*k**3/3*alphaR)
            alpha = alphaMW
        return alpha

class CalculateCrossSections:
    def __init__(self, centers, orient, all_radii, num, n, wp, eps_inf, gam_drude):
        """Defines the different system parameters.
        
        Keyword arguments:
        centers -- coordinates of centers of each prolate spheorid [cm]
        orient -- orientation of the long axis of prolate spheroid 
        all_radii -- radii of the oscillators [cm]
        num -- number of dipoles 
        """
        self.centers = centers
        self.orient = orient
        self.all_radii = all_radii
        self.num = num
        self.n = n
        self.wp = wp
        self.eps_inf = eps_inf
        self.gamNR_qs = gam_drude


    def A_ij(self, dip_i, dip_j, k):
        ''' off diagonal block terms in A_tilde '''
        A_ij = np.zeros( (3, 3) ,dtype=complex) 
        r_ij = self.centers[dip_i,:] - self.centers[dip_j,:] # [cm], distance between ith and jth dipole 
        r = np.sqrt(r_ij[0]**2+r_ij[1]**2+r_ij[2]**2)
        rx = r_ij[0]; ry = r_ij[1]; rz = r_ij[2]
        A_ij[0,0] = np.exp(1j*k*r)/r**3*(k**2*(rx*rx-r**2) + (1-1j*k*r)/r**2*(-3*rx*rx+r**2))
        A_ij[0,1] = np.exp(1j*k*r)/r**3*(k**2*(rx*ry) + (1-1j*k*r)/r**2*(-3*rx*ry))
        A_ij[0,2] = np.exp(1j*k*r)/r**3*(k**2*(rx*rz) + (1-1j*k*r)/r**2*(-3*rx*rz))
        A_ij[1,0] = A_ij[0,1]
        A_ij[1,1] = np.exp(1j*k*r)/r**3*(k**2*(ry*ry-r**2) + (1-1j*k*r)/r**2*(-3*ry*ry+r**2))
        A_ij[1,2] = np.exp(1j*k*r)/r**3*(k**2*(ry*rz) + (1-1j*k*r)/r**2*(-3*ry*rz))
        A_ij[2,0] = A_ij[0,2]
        A_ij[2,1] = A_ij[1,2]
        A_ij[2,2] = np.exp(1j*k*r)/r**3*(k**2*(rz*rz-r**2) + (1-1j*k*r)/r**2*(-3*rz*rz+r**2))
        return A_ij

    def alpha_tensor(self, dip_i, k):
        """ Polarizability (3x3 tensor) in the cartesian basis """
        alpha_rotated = np.zeros( (3, 3) ,dtype=complex)
        w = k*c
        dip_params = DipoleParameters(self.centers, self.orient, self.all_radii, self.n, self.wp, self.eps_inf, self.gamNR_qs)
        # alpha_rotated is the polarizability in a rotated frame, aligned with the long axis of the prolate spheroid
        alpha_rotated[0,0] = dip_params.alpha(w=w, radius=self.all_radii[dip_i, : ], kind='short')
        alpha_rotated[1,1] = dip_params.alpha(w=w, radius=self.all_radii[dip_i, : ], kind='short')
        alpha_rotated[2,2] = dip_params.alpha(w=w, radius=self.all_radii[dip_i, : ], kind='long')
        theta = self.orient[dip_i]
        R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
        invR = np.linalg.inv(R)
        alpha = np.matmul(R, np.matmul(alpha_rotated, invR))
        return alpha

    def A_ii(self, dip_i, dip_j, k):
        '''On diagonal block terms in A_tilde.'''
        w = k*c
        alpha = self.alpha_tensor(dip_i=dip_i, k=k)
        A_ii = np.linalg.inv(alpha)
        return A_ii

    def A_tilde(self, w):
        '''A_tilde = [3*N, 3*N]'''
        A_tilde = np.zeros( (3*self.num, 3*self.num) ,dtype=complex) 
        for i in range(0 , self.num): 
            for j in range(0, self.num):
                if i == j:  
                    A_tilde[3*i : 3*(i+1), 3*i : 3*(i+1)] = self.A_ii(dip_i=i, dip_j=i, k=w/c)
                if i != j:
                    A_tilde[3*i : 3*(i+1), 3*j : 3*(j+1)] = self.A_ij(dip_i=i, dip_j=j, k=w/c)
        return A_tilde

    def P_tilde(self, w, drive):
        E0_tilde = np.zeros((3*self.num, 1))
        P_tilde = np.zeros((3*self.num, 1),dtype=complex)
        for i in range(0, self.num):
            E0_tilde[3*i,:] = drive[0]
            E0_tilde[3*i+1,:] = drive[1]
            E0_tilde[3*i+2,:] = drive[2]
        P_tilde = np.linalg.solve(self.A_tilde(w), E0_tilde)
        return P_tilde

    def cross_sects(self, w, drive):
        ''' Works for spheres up to 50 nm radii, and in the window < 3.0 eV '''
        k = w/c
        P = self.P_tilde(w=w, drive=drive)
        P_each = np.zeros((3, self.num),dtype=complex)
        E_each = np.zeros((3, self.num),dtype=complex)
        Cext_each = np.zeros(self.num)
        Csca_each = np.zeros(self.num)

        for i in range(0, self.num):
            # Evaluate the cross sections of each particle separately
            dip_params = DipoleParameters(self.centers, self.orient, self.all_radii, self.n, self.wp, self.eps_inf, self.gamNR_qs)
            P_each[:,i] = P[3*i:3*(i+1), 0]
            E_each[0,i] = dip_params.alpha(w=w, radius=self.all_radii[i, :],kind='short')**(-1)*P_each[0,i]
            E_each[1,i] = dip_params.alpha(w=w, radius=self.all_radii[i, :],kind='short')**(-1)*P_each[1,i]
            E_each[2,i] = dip_params.alpha(w=w, radius=self.all_radii[i, :],kind='long')**(-1)*P_each[2,i]
            Cext_each[i] = 4*np.pi*k*np.imag( np.sum( P_each[:,i]*np.conj(E_each[:,i])) ) *10**8
            Csca_each[i] = 4*np.pi*k*2/3*k**3*np.real( np.sum(P_each[:,i]*np.conj(P_each[:,i])) ) *10**8  
        Cabs_each = Cext_each - Csca_each
        return Cext_each, Csca_each, Cabs_each

