from cosmosis.gaussian_likelihood import GaussianLikelihood
import numpy as np
import scipy.interpolate

# The three different types of measurement
# of BAO used in this data release
KIND_DV = 1
KIND_DM = 2
KIND_DH = 3


DESI_EXTENDED_DATA_SETS = {
    'BGS_1': {'kind': 'd_m_d_h', 'z_eff': 0.05, 'mean': [1.49, 21.56], 'sigma': [0.09, 2.77], 'corr': -0.45},
    'BGS_2': {'kind': 'd_m_d_h', 'z_eff': 0.15, 'mean': [4.31, 22.44], 'sigma': [0.10, 1.02], 'corr': -0.45},
    'BGS_3': {'kind': 'd_m_d_h', 'z_eff': 0.25, 'mean': [6.87, 19.66], 'sigma': [0.10, 0.61], 'corr': -0.45},
    'BGS_4': {'kind': 'd_m_d_h', 'z_eff': 0.35, 'mean': [9.59, 19.08], 'sigma': [0.12, 0.46], 'corr': -0.45},
    'LRG_1': {'kind': 'd_m_d_h', 'z_eff': 0.45, 'mean': [11.73, 17.87], 'sigma': [0.12, 0.36], 'corr': -0.45},
    'LRG_2': {'kind': 'd_m_d_h', 'z_eff': 0.55, 'mean': [13.97, 16.53], 'sigma': [0.12, 0.28], 'corr': -0.45},
    'LRG_3': {'kind': 'd_m_d_h', 'z_eff': 0.65, 'mean': [16.16, 15.24], 'sigma': [0.12, 0.24], 'corr': -0.45},
    'LRG_4': {'kind': 'd_m_d_h', 'z_eff': 0.75, 'mean': [17.90, 14.67], 'sigma': [0.13, 0.20], 'corr': -0.45},
    'LRG_5': {'kind': 'd_m_d_h', 'z_eff': 0.85, 'mean': [19.86, 13.98], 'sigma': [0.13, 0.17], 'corr': -0.45},
    'LRG_6': {'kind': 'd_m_d_h', 'z_eff': 0.95, 'mean': [21.62, 12.90], 'sigma': [0.15, 0.17], 'corr': -0.45},
    'LRG_7': {'kind': 'd_m_d_h', 'z_eff': 1.05, 'mean': [23.74, 12.35], 'sigma': [0.22, 0.19], 'corr': -0.45},
    'ELG_1': {'kind': 'd_m_d_h', 'z_eff': 1.15, 'mean': [25.01, 11.45], 'sigma': [0.26, 0.18], 'corr': -0.45},
    'ELG_2': {'kind': 'd_m_d_h', 'z_eff': 1.25, 'mean': [26.33, 11.09], 'sigma': [0.28, 0.17], 'corr': -0.45},
    'ELG_3': {'kind': 'd_m_d_h', 'z_eff': 1.35, 'mean': [27.54, 10.54], 'sigma': [0.30, 0.16], 'corr': -0.45},
    'ELG_4': {'kind': 'd_m_d_h', 'z_eff': 1.45, 'mean': [29.02, 9.89], 'sigma': [0.33, 0.16], 'corr': -0.45},
    'ELG_5': {'kind': 'd_m_d_h', 'z_eff': 1.55, 'mean': [30.20, 9.78], 'sigma': [0.46, 0.19], 'corr': -0.45},
    'QSO_1': {'kind': 'd_m_d_h', 'z_eff': 1.65, 'mean': [31.66, 8.55], 'sigma': [0.98, 0.39], 'corr': -0.45},
    'QSO_2': {'kind': 'd_m_d_h', 'z_eff': 1.75, 'mean': [33.67, 8.07], 'sigma': [1.03, 0.38], 'corr': -0.45},
    'QSO_3': {'kind': 'd_m_d_h', 'z_eff': 1.85, 'mean': [34.14, 7.38], 'sigma': [1.13, 0.38], 'corr': -0.45},
    'QSO_4': {'kind': 'd_m_d_h', 'z_eff': 1.95, 'mean': [33.32, 7.82], 'sigma': [1.22, 0.38], 'corr': -0.45},
    'QSO_5': {'kind': 'd_m_d_h', 'z_eff': 2.05, 'mean': [36.94, 7.47], 'sigma': [1.38, 0.40], 'corr': -0.45},
    'LYA_1': {'kind': 'd_m_d_h', 'z_eff': 2.15, 'mean': [36.79, 7.03], 'sigma': [0.67, 0.14], 'corr': -0.45},
    'LYA_2': {'kind': 'd_m_d_h', 'z_eff': 2.25, 'mean': [36.69, 6.67], 'sigma': [0.73, 0.14], 'corr': -0.45},
    'LYA_3': {'kind': 'd_m_d_h', 'z_eff': 2.35, 'mean': [38.26, 6.64], 'sigma': [0.82, 0.14], 'corr': -0.45},
    'LYA_4': {'kind': 'd_m_d_h', 'z_eff': 2.45, 'mean': [39.78, 5.97], 'sigma': [0.92, 0.14], 'corr': -0.45},
    'LYA_5': {'kind': 'd_m_d_h', 'z_eff': 2.55, 'mean': [40.61, 5.92], 'sigma': [1.06, 0.15], 'corr': -0.45},
    'LYA_6': {'kind': 'd_m_d_h', 'z_eff': 2.65, 'mean': [40.18, 5.84], 'sigma': [1.26, 0.16], 'corr': -0.45},
    'LYA_7': {'kind': 'd_m_d_h', 'z_eff': 2.75, 'mean': [43.30, 5.69], 'sigma': [1.50, 0.18], 'corr': -0.45},
    'LYA_8': {'kind': 'd_m_d_h', 'z_eff': 2.85, 'mean': [40.95, 5.26], 'sigma': [1.80, 0.20], 'corr': -0.45},
    'LYA_9': {'kind': 'd_m_d_h', 'z_eff': 2.95, 'mean': [43.88, 5.34], 'sigma': [2.19, 0.22], 'corr': -0.45},
    'LYA_10': {'kind': 'd_m_d_h', 'z_eff': 3.25, 'mean': [44.19, 4.58], 'sigma': [1.80, 0.15], 'corr': -0.45}
}

class DESIExtendedLikelihood(GaussianLikelihood):
    """
    DESI Extended Likelihood
    """
    # users can override this if they want to use a different name
    # which can be useful if you want to keep the different likelihoods
    # separately.
    like_name = "desi_bao"
    x_section = 'distances'
    x_name = 'z'
    y_section = 'distances'

    def __init__(self, options):
        data_sets = options.get_string("desi_data_sets")
        data_sets = data_sets.split(',')

        allowed = list(DESI_EXTENDED_DATA_SETS.keys())
        for data_set in data_sets:
            data_set = data_set.strip()
            if data_set not in allowed:
                raise ValueError(f"Unknown DESI-extended data set {data_set}. Valid options are: {allowed} (comma-separated to use more than one)")
        self.data_sets = data_sets
        super().__init__(options)
    

    def build_data(self):
        z = []
        mu = []
        kinds = []
        for name in self.data_sets:
            ds = DESI_EXTENDED_DATA_SETS[name]

            # collect the effective redshfits for the measurements
            z.append(ds["z_eff"])

            # The d_v type measurements are just a single number
            # but the d_m_d_h measurements are two values
            if ds["kind"] == "d_v":
                mu.append(ds["mean"])
                kinds.append(KIND_DV)
            else:
                mu.extend(ds["mean"])
                kinds.append(KIND_DM)
                kinds.append(KIND_DH)
                # This makes the z array the same length
                # as the mu array. But because the D_M and D_H
                # measurements are at the same redshift we only
                # need to store the redshift once, and this should
                # hopefully trigger an error if we mess up later.
                z.append(-1.0)

        kinds = np.array(kinds)
        z = np.array(z)
        mu = np.array(mu)

        # record the indices of the d_v and d_m_d_h measurements
        # for later
        self.dv_index = np.where(kinds==KIND_DV)[0]
        self.dm_index = np.where(kinds==KIND_DM)[0]
        self.dh_index = np.where(kinds==KIND_DH)[0]

        self.any_dv = len(self.dv_index) > 0
        self.any_dmdh = len(self.dm_index) > 0

        return z, mu

    def build_covariance(self):
        n = len(self.data_x)
        C = np.zeros((n, n))
        i = 0
        for name in self.data_sets:
            ds = DESI_EXTENDED_DATA_SETS[name]
            if ds["kind"] == "d_v":
                C[i, i] = ds["sigma"]**2
                i += 1
            else:
                C[i, i] = ds["sigma"][0]**2
                C[i+1, i+1] = ds["sigma"][1]**2
                C[i, i+1] = C[i+1, i] = ds["corr"]*ds["sigma"][0]*ds["sigma"][1]
                i += 2
        return C

    def extract_theory_points(self, block):
        z_theory = block[self.x_section, self.x_name]
        y = np.zeros(self.data_x.size)
        r_s = block[self.y_section, "rs_zdrag"]

        block["distances", "h0rd"] = block["cosmological_parameters", "h0"] * r_s

        if self.any_dv:
            d_v = block[self.y_section, "d_v"]
            z_data = self.data_x[self.dv_index]
            f = scipy.interpolate.interp1d(z_theory, d_v/r_s, kind=self.kind)
            y[self.dv_index] = f(z_data)

        if self.any_dmdh:
            z_data = self.data_x[self.dm_index]

            d_m = block[self.y_section, "d_m"]
            f = scipy.interpolate.interp1d(z_theory, d_m/r_s, kind=self.kind)
            y[self.dm_index] = f(z_data)

            d_h = 1.0 / block[self.y_section, "h"]
            f = scipy.interpolate.interp1d(z_theory, d_h/r_s, kind=self.kind)
            y[self.dh_index] = f(z_data)
        return y

setup, execute, cleanup = DESIExtendedLikelihood.build_module()
