from cosmosis.gaussian_likelihood import GaussianLikelihood
import numpy as np
import scipy.interpolate

# The three different types of measurement
# of BAO used in this data release
KIND_DV = 1
KIND_DM = 2
KIND_DH = 3


DESI_EXTENDED_DATA_SETS = {'BGS_1': {'kind': 'd_m_d_h',
  'z_eff': 0.05,
  'mean': [1.4483710776213172, 21.94602323377216],
  'sigma': [0.08733677598056543, 2.7717827344254236],
  'corr': -0.45},
 'BGS_2': {'kind': 'd_m_d_h',
  'z_eff': 0.15000000000000002,
  'mean': [4.242753250008926, 20.88625491576449],
  'sigma': [0.09885615072520798, 1.02342649087246],
  'corr': -0.45},
 'BGS_3': {'kind': 'd_m_d_h',
  'z_eff': 0.25,
  'mean': [6.897186949020114, 19.805183200027507],
  'sigma': [0.10276808554039968, 0.6119801608808499],
  'corr': -0.45},
 'BGS_4': {'kind': 'd_m_d_h',
  'z_eff': 0.35,
  'mean': [9.410590095895312, 18.72804154336819],
  'sigma': [0.11669131718910186, 0.4588370178125206],
  'corr': -0.45},
 'LRG_1': {'kind': 'd_m_d_h',
  'z_eff': 0.45,
  'mean': [11.784845997981085, 17.67494217729097],
  'sigma': [0.12256239837900329, 0.36056882041673577],
  'corr': -0.45},
 'LRG_2': {'kind': 'd_m_d_h',
  'z_eff': 0.55,
  'mean': [14.024110992604896, 16.66064887584476],
  'sigma': [0.12341217673492309, 0.2848970957769454],
  'corr': -0.45},
 'LRG_3': {'kind': 'd_m_d_h',
  'z_eff': 0.6499999999999999,
  'mean': [16.134140987208212, 15.695033874725672],
  'sigma': [0.12423288560150324, 0.23542550812088506],
  'corr': -0.45},
 'LRG_4': {'kind': 'd_m_d_h',
  'z_eff': 0.75,
  'mean': [18.12170363227118, 14.78388358230321],
  'sigma': [0.12685192542589827, 0.1966256516446327],
  'corr': -0.45},
 'LRG_5': {'kind': 'd_m_d_h',
  'z_eff': 0.8500000000000001,
  'mean': [19.99410413553156, 13.929801505674162],
  'sigma': [0.12996167688095514, 0.17133655851979218],
  'corr': -0.45},
 'LRG_6': {'kind': 'd_m_d_h',
  'z_eff': 0.95,
  'mean': [21.758826394680167, 13.133055095985632],
  'sigma': [0.15231178476276117, 0.16547649420941898],
  'corr': -0.45},
 'LRG_7': {'kind': 'd_m_d_h',
  'z_eff': 1.05,
  'mean': [23.423276630084075, 12.392293260086578],
  'sigma': [0.21783647265978193, 0.19208054553134196],
  'corr': -0.45},
 'ELG_1': {'kind': 'd_m_d_h',
  'z_eff': 1.15,
  'mean': [24.994610883946688, 11.705114192811966],
  'sigma': [0.25744449210465087, 0.17791773573074188],
  'corr': -0.45},
 'ELG_2': {'kind': 'd_m_d_h',
  'z_eff': 1.25,
  'mean': [26.479627224527373, 11.068492416843103],
  'sigma': [0.2753881231350847, 0.16824108473601515],
  'corr': -0.45},
 'ELG_3': {'kind': 'd_m_d_h',
  'z_eff': 1.35,
  'mean': [27.884705627428605, 10.479086719857081],
  'sigma': [0.2983663502134861, 0.16137793548579907],
  'corr': -0.45},
 'ELG_4': {'kind': 'd_m_d_h',
  'z_eff': 1.45,
  'mean': [29.21578161252514, 9.933453768083675],
  'sigma': [0.3301383322215341, 0.15794191491253043],
  'corr': -0.45},
 'ELG_5': {'kind': 'd_m_d_h',
  'z_eff': 1.55,
  'mean': [30.47834286377178, 9.428190366628973],
  'sigma': [0.46022297724295386, 0.18856380733257946],
  'corr': -0.45},
 'QSO_1': {'kind': 'd_m_d_h',
  'z_eff': 1.65,
  'mean': [31.67744082155518, 8.960023587195245],
  'sigma': [0.9756651773038996, 0.3870730189668346],
  'corr': -0.45},
 'QSO_2': {'kind': 'd_m_d_h',
  'z_eff': 1.75,
  'mean': [32.81771147614082, 8.525863888367356],
  'sigma': [1.0337579114984359, 0.3768431838658371],
  'corr': -0.45},
 'QSO_3': {'kind': 'd_m_d_h',
  'z_eff': 1.85,
  'mean': [33.90340131619271, 8.122832651699332],
  'sigma': [1.1289832638292174, 0.3785240015691889],
  'corr': -0.45},
 'QSO_4': {'kind': 'd_m_d_h',
  'z_eff': 1.95,
  'mean': [34.938395670250536, 7.748272507992089],
  'sigma': [1.2158561693247187, 0.3765660438884156],
  'corr': -0.45},
 'QSO_5': {'kind': 'd_m_d_h',
  'z_eff': 2.05,
  'mean': [35.92624760962466, 7.399746452194654],
  'sigma': [1.3759752834486243, 0.39588643519241395],
  'corr': -0.45},
 'LYA_1': {'kind': 'd_m_d_h',
  'z_eff': 2.1500000000000004,
  'mean': [36.870206241448884, 7.075029959894756],
  'sigma': [0.6747247742185146, 0.1386705872139372],
  'corr': -0.45},
 'LYA_2': {'kind': 'd_m_d_h',
  'z_eff': 2.25,
  'mean': [37.773243680385605, 6.7720990106510035],
  'sigma': [0.7328009273994808, 0.13747360991621535],
  'corr': -0.45},
 'LYA_3': {'kind': 'd_m_d_h',
  'z_eff': 2.3499999999999996,
  'mean': [38.63808030176275, 6.4891159839272285],
  'sigma': [0.815263494367194, 0.1388670820560427],
  'corr': -0.45},
 'LYA_4': {'kind': 'd_m_d_h',
  'z_eff': 2.45,
  'mean': [39.46720808947964, 6.2244147288201175],
  'sigma': [0.9156392276759275, 0.14253909728998068],
  'corr': -0.45},
 'LYA_5': {'kind': 'd_m_d_h',
  'z_eff': 2.55,
  'mean': [40.26291202967843, 5.976485644965744],
  'sigma': [1.0589145863805427, 0.15000978968864018],
  'corr': -0.45},
 'LYA_6': {'kind': 'd_m_d_h',
  'z_eff': 2.6500000000000004,
  'mean': [41.02728958826514, 5.743961292450113],
  'sigma': [1.2554350614009133, 0.16197970844709317],
  'corr': -0.45},
 'LYA_7': {'kind': 'd_m_d_h',
  'z_eff': 2.75,
  'mean': [41.762268362823185, 5.5256028313263315],
  'sigma': [1.49508920738907, 0.17571417003617737],
  'corr': -0.45},
 'LYA_8': {'kind': 'd_m_d_h',
  'z_eff': 2.8499999999999996,
  'mean': [42.46962202855099, 5.320287445972811],
  'sigma': [1.8049589362134169, 0.19578657801179947],
  'corr': -0.45},
 'LYA_9': {'kind': 'd_m_d_h',
  'z_eff': 2.95,
  'mean': [43.150984711435655, 5.1269968141416715],
  'sigma': [2.187754924869788, 0.21892276396384935],
  'corr': -0.45},
 'LYA_10': {'kind': 'd_m_d_h',
  'z_eff': 3.25,
  'mean': [45.05363757769382, 4.610444359430172],
  'sigma': [1.802145503107753, 0.14661213062987946],
  'corr': -0.45}}

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
