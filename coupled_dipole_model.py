import numpy as np
from scipy.special import spherical_jn as sph_jn
from scipy.special import spherical_yn as sph_yn

e = 4.80326E-10  # elementary charge, [statC, g^1/2 cm^3/2 s^-1]
c = 2.998E+10  # speed of light [cm/s]
hbar_eVs = 6.58212E-16  # Planck's constant [eV*s]


class Sphere_Polarizability:
    """ Defines polarizability of spherical particles using Mie.
    """

    def __init__(self,
                 all_radii,  # radii of the spheres [cm]
                 n,  # refractive index of background [unitless]
                 wp,  # wp -- bulk plasma frequency [1/s]
                 eps_inf,  # response of ionic background [unitless]
                 gam_drude,  # Drude damping (quasistatic, nonrad.) [1/s]
                 shell=0,
                 shell_m=0):
        """Defines the different system parameters.
        """

        self.all_radii = all_radii
        self.n = n
        self.wp = wp
        self.eps_inf = eps_inf
        self.gamNR_qs = gam_drude
        self.shell = shell
        self.shell_m = shell_m

    def psi(self, rho):
        return rho * sph_jn(1, rho)

    def psi_p(self, rho):
        return sph_jn(1, rho) + rho * sph_jn(1, rho, derivative=True)

    def hankel(self, rho):
        return sph_jn(1, rho) + 1j * sph_yn(1, rho)

    def hankel_p(self, rho):
        return (sph_jn(1, rho, derivative=True)
                + 1j * sph_yn(1, rho, derivative=True))

    def xi(self, rho):
        return rho * self.hankel(rho)

    def xi_p(self, rho):
        return self.hankel(rho) + rho * self.hankel_p(rho)

    def optical_params(self, w, radius):
        eps = self.eps_inf - self.wp**2 / (w**2 + 1j * w * self.gamNR_qs)
        m1 = np.sqrt(eps)
        m2 = self.shell_m
        k = w / c * self.n
        x = k * radius
        y = k * (radius + self.shell)
        return m1, m2, k, x, y

    def mie_coefficents(self, w, radius):
        m, _, k, x, _ = self.optical_params(w, radius)
        numer_a = (m * self.psi(m * x) * self.psi_p(x)
                   - self.psi(x) * self.psi_p(m * x))
        denom_a = (m * self.psi(m * x) * self.xi_p(x)
                   - self.xi(x) * self.psi_p(m * x))
        numer_b = (self.psi(m * x) * self.psi_p(x)
                   - m * self.psi(x) * self.psi_p(m * x))
        denom_b = (self.psi(m * x) * self.xi_p(x)
                   - m * self.xi(x) * self.psi_p(m * x))
        a = numer_a / denom_a
        b = numer_b / denom_b
        return a, b

    def chi(self, z):
        return -z * sph_yn(1, z)

    def chi_p(self, z):
        return -sph_yn(1, z) + -z * sph_yn(1, z, derivative=True)

    def mie_coeff_coated_sphere(self, w, radius):
        m1, m2, k, x, y = self.optical_params(w, radius)
        A_num = (m2 * self.psi(m2 * x) * self.psi_p(m1 * x)
                 - m1 * self.psi_p(m2 * x) * self.psi(m1 * x))
        A_denom = (m2 * self.chi(m2 * x) * self.psi_p(m1 * x)
                   - m1 * self.chi_p(m2 * x) * self.psi(m1 * x))
        A = A_num / A_denom
        a_num = (self.psi(y) * (self.psi_p(m2 * y) - A * self.chi_p(m2 * y))
                 - (m2 * self.psi_p(y)
                     * (self.psi(m2 * y) - A * self.chi(m2 * y))))

        a_den = ((self.xi(y) * (self.psi_p(m2 * y) - A * self.chi_p(m2 * y))
                  - (m2 * self.xi_p(y)
                     * (self.psi(m2 * y) - A * self.chi(m2 * y)))))
        return a_num / a_den

    def alpha(self, w, radius):
        """ Polarizability [cm^3]
        """
        k = w / c * self.n
        if self.shell == 0:
            a, _ = self.mie_coefficents(w=w, radius=radius)
        if self.shell != 0:
            a = self.mie_coeff_coated_sphere(w=w, radius=radius)
        alpha = 3 / (2. * k**3) * 1j * a
        return alpha


class CrossSections:
    def __init__(self,
                 alphas,
                 centers,
                 all_radii,
                 unit_vecs,
                 w,
                 n,
                 ):
        self.alphas = alphas
        self.centers = centers  # [num_part, dim]
        self.all_radii = all_radii
        self.unit_vecs = unit_vecs
        self.w = w  # [1/s]
        self.num = len(alphas)  # number of particles
        self.n = n
        self.k = self.w / c * self.n  # [1/cm]

    def A_ii(self, dip_i, dip_j):
        """On diagonal vectors of A_matrix.
           Alpha^-1 p_hat
        """
        A_ii = self.alphas[dip_i]**(-1)
        # print("{:.1e}".format(A_ii))
        return A_ii

    def A_ij(self, dip_i, dip_j):
        """Off diagonal vectors of A_matrix
        """
        r_ij = self.centers[dip_i, :] - self.centers[dip_j, :]
        phat_i = self.unit_vecs[dip_i, :]
        phat_j = self.unit_vecs[dip_j, :]
        magr_ij = np.linalg.norm(r_ij)
        if magr_ij == 0:
            A_ij = 0
            return A_ij

        rhat_ij = r_ij / magr_ij
        phat_i_dot_nn_dot_phat_j = (np.dot(phat_i, rhat_ij)
                                    * np.dot(rhat_ij, phat_j))
        near = ((3 * phat_i_dot_nn_dot_phat_j - np.dot(phat_i, phat_j))
                / magr_ij**3)
        intermed = 0#(-1j * self.k
                    #* (3 * phat_i_dot_nn_dot_phat_j - np.dot(phat_i, phat_j))
                    #/ magr_ij**2)
        far = 0#(self.k**2
               #* (phat_i_dot_nn_dot_phat_j - np.dot(phat_i, phat_j))
               #/ magr_ij)
        # print("{:.1e}, {:.1e}, {:.1e}".format(near, intermed, far))
        # print()
        A_ij = np.exp(1j * self.k * magr_ij) * (near + intermed + far)
        return A_ij

    def A_Matrix(self):
        """A_Matrix = [N, N]
        """
        A_Matrix = np.zeros((self.num,
                             self.num),
                            dtype=complex)
        for i in range(0, self.num):
            for j in range(0, self.num):
                if i == j:
                    A_Matrix[i, j] = self.A_ii(dip_i=i, dip_j=j)
                if i != j:
                    A_Matrix[i, j] = -self.A_ij(dip_i=i, dip_j=j)
        return A_Matrix

    def P_Mags(self, drive):
        Einc_Vecs = np.tile(drive, (self.num, 1))
        A_Matrix = self.A_Matrix()
        B = np.sum(self.unit_vecs
                   * Einc_Vecs, axis=1)[:, np.newaxis]
        P_Mags = np.linalg.solve(A_Matrix, B)
        return P_Mags

    def cross_sects(self, drive):
        P = (self.P_Mags(drive=drive) * self.unit_vecs)
        # print(self.alphas.reshape(1, self.num).shape, P.shape)
        Cext = np.zeros((self.num))
        for i in range(self.num):
            E = self.alphas[i]**(-1) * P[i, :]
            Cext[i] = ((4 * np.pi * self.k
                        * np.imag(np.dot(P[i, :], np.conj(E))) * 10**8))
        # Cabs_ea = (Cext_ea - 4 * np.pi * self.k * 2 / 3 * self.k**3
        #            * np.real(np.sum(P_ea * np.conj(P_ea), axis=2)) * 10**8)
        return Cext  # , Cabs_ea
