import numpy as np
from scipy.special import spherical_jn as sph_jn
from scipy.special import spherical_yn as sph_yn

e = 4.80326E-10  # elementary charge, [statC, g^1/2 cm^3/2 s^-1]
c = 2.998E+10  # speed of light [cm/s]
hbar_eVs = 6.58212E-16  # Planck's constant [eV*s]


class Polarizability:
    """ Defines various electric polarizabilities.
    """
    def __init__(self,
                 nback,          # refractive index of background [unitless]
                ):
        """Defines the different system parameters.
        """
        self.nback = nback

    def psi(self, n, rho):
        return rho * sph_jn(n, rho)

    def psi_p(self, n, rho):
        return sph_jn(n, rho) + rho * sph_jn(n, rho, derivative=True)

    def hankel(self, n, rho):
        return sph_jn(n, rho) + 1j * sph_yn(n, rho)

    def hankel_p(self, n, rho):
        return (sph_jn(n, rho, derivative=True)
                + 1j * sph_yn(n, rho, derivative=True))

    def xi(self, n, rho):
        return rho * self.hankel(n, rho)

    def xi_p(self, n, rho):
        return self.hankel(n, rho) + rho * self.hankel_p(n, rho)

    def drude_model(self, w, eps_inf, wp, gamNR):
        eps = eps_inf - wp**2 / (w**2 + 1j * w * gamNR)
        return eps
    
    def dielectric_data(self):
        data = np.loadtxt('auJC_interp.tab', skiprows=3)
        wave_raw = data[:, 0] * 1E-4  # um -> cm
        w_raw = 2 * np.pi * c / wave_raw  # rad/s
        n_diel = data[:, 1]
        k_diel = data[:, 2]
        eps_r = n_diel**2 - k_diel**2
        eps_im = 2 * n_diel * k_diel
        return w_raw, eps_r + 1j * eps_im

    def mie_coefficents(self, eps, w, radius):
        n = 1 # multipole order
        m = np.sqrt(eps) / self.nback
        k = w / c * self.nback
        x = k * radius
        numer_a = (m * self.psi(n, m * x) * self.psi_p(n, x)
                   - self.psi(n, x) * self.psi_p(n, m * x))
        denom_a = (m * self.psi(n, m * x) * self.xi_p(n, x)
                   - self.xi(n, x) * self.psi_p(n, m * x))
        numer_b = (self.psi(n, m * x) * self.psi_p(n, x)
                   - m * self.psi(n, x) * self.psi_p(n, m * x))
        denom_b = (self.psi(n, m * x) * self.xi_p(n, x)
                   - m * self.xi(n, x) * self.psi_p(n, m * x))
        a = numer_a / denom_a
        b = numer_b / denom_b
        return a, b

    def alpha_Mie(self, eps, w, radius):
        """ Polarizability [cm^3]
        """
        k = w / c * self.nback
        a, _ = self.mie_coefficents(eps, w, radius)
        alpha = 3 / (2. * k**3) * 1j * (a)  # + b)
        return alpha

    def alpha_Mie_coated(self, w, radius):
        """ Polarizability [cm^3]
        """
        k = w / c * self.nback
        a = self.mie_coeff_coated_sphere(w=w, radius=radius)
        alpha = 3 / (2. * k**3) * 1j * (a)  # + b)
        return alpha

    def spheroid_params(self, ai, ci):
        if ai > ci: # Oblate
            es = np.sqrt((ai**2 - ci**2) / ai**2)
            Lz = 1 / es**2 * (1 - np.sqrt(1 - es**2) / es * np.arcsin(es))
            Lxy = (1 - Lz) / 2   # ← sum rule, same pattern as prolate branch
            Dz = 3 / 4 * ((1 - 2 * es**2) * Lz + 1)
            Dxy = (3 * np.sqrt(1-es**2) / es * np.arctan(es) - Dz) * ai / (2 * ci)
        if ai < ci: # Prolate
            es = np.sqrt((ci**2 - ai**2) / ci**2)
            Lz = (1 - es**2) / es**3 * (-es + 1 / 2 * np.log((1 + es) / (1 - es)))
            Lxy = (1 - Lz) / 2
            Dz = 3 / 4 * ((1 + es**2) / (1 - es**2) * Lz + 1)
            Dxy = (3 / es * np.arctanh(es) - Dz) * ai / (2 * ci)
        V = 4 / 3 * np.pi * ai**2 * ci
        return V, es, Lz, Lxy, Dz, Dxy

    def alpha_spheroid_CS_MW(self, eps, w, ai, ci, which):
        eps_back = self.nback**2
        k_m = w / c * self.nback  # wave vector in medium
        V, es, Lz, Lxy, Dz, Dxy = self.spheroid_params(ai, ci)
        if which == 'z': # rotational axis of spheroid
            Lm = Lz
            lE = ci
            D = Dz
        else:
            Lm = Lxy
            lE = ai
            D = Dxy
        qm = (Lm - k_m**2 * V / (4 * np.pi * lE) * D
              - 1j * 2 * k_m**3 / 3 * V / (4 * np.pi))
        return (V / (4 * np.pi) * (eps - eps_back)
                / (eps_back + qm * (eps - eps_back)))


class CrossSections:
    def __init__(self,
                 alphas,
                 centers,
                 unit_vecs,
                 w,
                 n,
                 ):
        self.alphas = alphas
        self.centers = centers  # [num_part, dim]
        self.unit_vecs = unit_vecs
        self.w = w  # [1/s]
        self.num = len(alphas)  # number of particles
        self.n = n
        self.k = self.w / c * self.n  # [1/cm]

    def A_ii(self, dip_i, dip_j):
        """On diagonal vectors of A_matrix.
           Alpha^-1
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
        near = ((3. * phat_i_dot_nn_dot_phat_j - np.dot(phat_i, phat_j))
                / magr_ij**3)
        intermed = (-1j * self.k
                    * (3 * phat_i_dot_nn_dot_phat_j - np.dot(phat_i, phat_j))
                    / magr_ij**2) * 0
        far = (self.k**2
               * (np.dot(phat_i, phat_j) - phat_i_dot_nn_dot_phat_j)
               / magr_ij) * 0
        # print("{:.1e}, {:.1e}, {:.1e}".format(near, intermed, far))
        # print()
        # A_ij = np.exp(1j * self.k * magr_ij) * (near + intermed + far)
        A_ij = (near + intermed + far)


        return A_ij
    

    def A_Matrix(self):
        """A_Matrix = [N, N] — vectorized over all pairs."""
        N = self.num
        # pairwise displacements (N, N, 3) and distances (N, N)
        r_ij   = self.centers[:, np.newaxis, :] - self.centers[np.newaxis, :, :]
        mag_r  = np.linalg.norm(r_ij, axis=-1)
        safe_r = np.where(mag_r == 0, 1.0, mag_r)
        r_hat  = r_ij / safe_r[:, :, np.newaxis]

        pi = self.unit_vecs[:, np.newaxis, :]   # (N, 1, 3)
        pj = self.unit_vecs[np.newaxis, :, :]   # (1, N, 3)

        pi_dot_pj  = np.sum(pi * pj,    axis=-1)          # (N, N)
        pi_dot_r   = np.sum(pi * r_hat, axis=-1)          # (N, N)
        r_dot_pj   = np.sum(r_hat * pj, axis=-1)          # (N, N)
        pi_rr_pj   = pi_dot_r * r_dot_pj                  # (N, N)

        near   = (3.  * pi_rr_pj - pi_dot_pj) / safe_r**3
        intermed = (-1j * self.k * (3 * pi_rr_pj - pi_dot_pj)) / safe_r**2
        far    = self.k**2 * (pi_dot_pj - pi_rr_pj) / safe_r

        off_diag = np.exp(1j * self.k * safe_r) * (near + intermed + far)
        off_diag[np.arange(N), np.arange(N)] = 0.0   # zero out self-terms

        A = -off_diag
        np.fill_diagonal(A, np.array([self.alphas[i]**(-1) for i in range(N)]))
        return A

        # ── old loop version (kept for reference) ──────────────────────────
        # A_Matrix = np.zeros((self.num, self.num), dtype=complex)
        # for i in range(0, self.num):
        #     for j in range(0, self.num):
        #         if i == j:
        #             A_Matrix[i, j] = self.A_ii(dip_i=i, dip_j=j)
        #         if i != j:
        #             A_Matrix[i, j] = -self.A_ij(dip_i=i, dip_j=j)
        # return A_Matrix

    def P_Mags(self, drive, direction=None, gauss_env=None):
        if direction is not None:
            # Phase factor exp(ik * dir . r_i) for each dipole's position
            phases = np.exp(1j * self.k * (self.centers @ direction))  # (num,)
            Einc_Vecs = drive[np.newaxis, :] * phases[:, np.newaxis]   # (num, 3)
        else:
            Einc_Vecs = np.tile(drive, (self.num, 1))
        if gauss_env is not None:
            Einc_Vecs = Einc_Vecs * gauss_env[:, np.newaxis]   # per-dipole amplitude
        A_Matrix = self.A_Matrix()
        B = np.sum(self.unit_vecs
                   * Einc_Vecs, axis=1)[:, np.newaxis]
        P_Mags = np.linalg.solve(A_Matrix, B)
        return P_Mags

    def cross_sects(self, drive, direction=None, gauss_env=None):
        P = (self.P_Mags(drive=drive, direction=direction, gauss_env=gauss_env)
             * self.unit_vecs)
        if direction is not None:
            phases = np.exp(1j * self.k * (self.centers @ direction))
            Einc = drive[np.newaxis, :] * phases[:, np.newaxis]
        else:
            Einc = np.tile(drive, (self.num, 1))
        if gauss_env is not None:
            Einc = Einc * gauss_env[:, np.newaxis]

        Cext = (4 * np.pi * self.k
                * np.imag(np.sum(np.conj(Einc) * P, axis=-1)) * 1e8)
        p_sq      = np.real(np.sum(P * np.conj(P), axis=-1))
        alpha_inv = np.array([self.alphas[i]**(-1) for i in range(self.num)])
        Cabs = (4 * np.pi * self.k
                * (-np.imag(alpha_inv) - 2/3 * self.k**3)
                * p_sq * 1e8)
        return Cext, Cabs

        # ── old loop version (kept for reference) ──────────────────────────
        # Cext = np.zeros((self.num))
        # Cabs = np.zeros((self.num))
        # for i in range(self.num):
        #     Cext[i] = (4 * np.pi * self.k
        #                * np.imag(np.dot(np.conj(Einc[i, :]), P[i, :])) * 10**8)
        #     p_sq = np.real(np.dot(P[i, :], np.conj(P[i, :])))
        #     alpha_inv = self.alphas[i]**(-1)
        #     Cabs[i] = (4 * np.pi * self.k
        #                * (-np.imag(alpha_inv) - 2/3 * self.k**3)
        #                * p_sq * 10**8)
        # return Cext, Cabs
