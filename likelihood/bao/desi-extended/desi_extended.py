from cosmosis.gaussian_likelihood import GaussianLikelihood
import numpy as np
import scipy.interpolate

# The three different types of measurement
# of BAO used in this data release
KIND_DV = 1
KIND_DM = 2
KIND_DH = 3

# no error added
# fiducial cosmology values
# H0_true = 66.7
# Om0_true = 0.318
# w0_true = -0.908
# alpha_true = 1.45
# rd_true = 100 / (H0_true / 100)

DESI_EXTENDED_DATA_SETS = {
    "BGS_1": {
        "kind": "d_m_d_h",
        "z_eff": 0.05,
        "mean": [1.4777008454892422, 29.13155673100815],
        "sigma": [0.08925709804968576, 3.7488885661659976],
        "corr": 0,
    },
    "BGS_2": {
        "kind": "d_m_d_h",
        "z_eff": 0.15,
        "mean": [4.307981376238263, 27.48472225150984],
        "sigma": [0.09995316418186226, 1.2578390647890434],
        "corr": 0,
    },
    "BGS_3": {
        "kind": "d_m_d_h",
        "z_eff": 0.25,
        "mean": [6.97680842704454, 25.902940504114763],
        "sigma": [0.10155470781724221, 0.8078889825844762],
        "corr": 0,
    },
    "BGS_4": {
        "kind": "d_m_d_h",
        "z_eff": 0.35,
        "mean": [9.490881177618657, 24.390540672743395],
        "sigma": [0.11875972276477986, 0.587960000146486],
        "corr": 0,
    },
    "LRG_1": {
        "kind": "d_m_d_h",
        "z_eff": 0.45,
        "mean": [11.85741286048057, 22.952992943888233],
        "sigma": [0.12130345637320275, 0.462801659529278],
        "corr": 0,
    },
    "LRG_2": {
        "kind": "d_m_d_h",
        "z_eff": 0.55,
        "mean": [14.084127173813114, 21.594802399848266],
        "sigma": [0.12098033363332666, 0.3705972582534257],
        "corr": 0,
    },
    "LRG_3": {
        "kind": "d_m_d_h",
        "z_eff": 0.65,
        "mean": [16.179108844830875, 20.31860762629913],
        "sigma": [0.12014189736260548, 0.31668015908259073],
        "corr": 0,
    },
    "LRG_4": {
        "kind": "d_m_d_h",
        "z_eff": 0.75,
        "mean": [18.150603997135917, 19.12501063718247],
        "sigma": [0.13182002902947876, 0.25979638274124567],
        "corr": 0,
    },
    "LRG_5": {
        "kind": "d_m_d_h",
        "z_eff": 0.85,
        "mean": [20.006825330882293, 18.012796549255757],
        "sigma": [0.1309610922968126, 0.21725960750198833],
        "corr": 0,
    },
    "LRG_6": {
        "kind": "d_m_d_h",
        "z_eff": 0.95,
        "mean": [21.755788922704866, 16.979317105026755],
        "sigma": [0.15094210630923818, 0.2219518575820491],
        "corr": 0,
    },
    "LRG_7": {
        "kind": "d_m_d_h",
        "z_eff": 1.05,
        "mean": [23.405191663710152, 16.02090372661127],
        "sigma": [0.2168973111211556, 0.24861737626646913],
        "corr": 0,
    },
    "ELG_1": {
        "kind": "d_m_d_h",
        "z_eff": 1.15,
        "mean": [24.962328381980385, 15.13324065892713],
        "sigma": [0.25950441340723307, 0.2329749231294003],
        "corr": 0,
    },
    "ELG_2": {
        "kind": "d_m_d_h",
        "z_eff": 1.25,
        "mean": [26.434042990987226, 14.311670063977589],
        "sigma": [0.28110641995732716, 0.21774325131916109],
        "corr": 0,
    },
    "ELG_3": {
        "kind": "d_m_d_h",
        "z_eff": 1.35,
        "mean": [27.826706490463284, 13.55142446052109],
        "sigma": [0.3031231643841316, 0.20696720994250392],
        "corr": 0,
    },
    "ELG_4": {
        "kind": "d_m_d_h",
        "z_eff": 1.45,
        "mean": [29.146214845153438, 12.847793548093254],
        "sigma": [0.33143524806687236, 0.20915012752709947],
        "corr": 0,
    },
    "ELG_5": {
        "kind": "d_m_d_h",
        "z_eff": 1.55,
        "mean": [30.39800073833833, 12.196236926000651],
        "sigma": [0.4630159052859481, 0.2389544852272855],
        "corr": 0,
    },
    "QSO_1": {
        "kind": "d_m_d_h",
        "z_eff": 1.65,
        "mean": [31.587054399326686, 11.592454797890817],
        "sigma": [0.9777420502634286, 0.519375214959266],
        "corr": 0,
    },
    "QSO_2": {
        "kind": "d_m_d_h",
        "z_eff": 1.75,
        "mean": [32.71794985977245, 11.032427498589733],
        "sigma": [1.0008758050361042, 0.5138678207518032],
        "corr": 0,
    },
    "QSO_3": {
        "kind": "d_m_d_h",
        "z_eff": 1.85,
        "mean": [33.7948739840746, 10.512432769092214],
        "sigma": [1.1185766725836055, 0.5349005251147648],
        "corr": 0,
    },
    "QSO_4": {
        "kind": "d_m_d_h",
        "z_eff": 1.95,
        "mean": [34.82165641265569, 10.029047746787851],
        "sigma": [1.2749826177502985, 0.4813157096891329],
        "corr": 0,
    },
    "QSO_5": {
        "kind": "d_m_d_h",
        "z_eff": 2.05,
        "mean": [35.80179916447854, 9.579140903867557],
        "sigma": [1.3374792324575089, 0.5114120400422105],
        "corr": 0,
    },
    "LYA_1": {
        "kind": "d_m_d_h",
        "z_eff": 2.15,
        "mean": [36.73850509474564, 9.1598577466563],
        "sigma": [0.6690622020516331, 0.1796050538560059],
        "corr": 0,
    },
    "LYA_2": {
        "kind": "d_m_d_h",
        "z_eff": 2.25,
        "mean": [37.63470472714522, 8.768602980875198],
        "sigma": [0.7487962510443175, 0.18121108341647937],
        "corr": 0,
    },
    "LYA_3": {
        "kind": "d_m_d_h",
        "z_eff": 2.35,
        "mean": [38.49308120668208, 8.403021012564624],
        "sigma": [0.8249954675765631, 0.17465863536508455],
        "corr": 0,
    },
    "LYA_4": {
        "kind": "d_m_d_h",
        "z_eff": 2.45,
        "mean": [39.316093273777774, 8.060976040461128],
        "sigma": [0.9092711365479023, 0.1966091717185641],
        "corr": 0,
    },
    "LYA_5": {
        "kind": "d_m_d_h",
        "z_eff": 2.55,
        "mean": [40.105996262562876, 7.74053255369923],
        "sigma": [1.0468445219974551, 0.20053193144298526],
        "corr": 0,
    },
    "LYA_6": {
        "kind": "d_m_d_h",
        "z_eff": 2.65,
        "mean": [40.86486119113433, 7.439936736498869],
        "sigma": [1.281476483345676, 0.20503762659642552],
        "corr": 0,
    },
    "LYA_7": {
        "kind": "d_m_d_h",
        "z_eff": 2.75,
        "mean": [41.594592050212896, 7.1575990649538985],
        "sigma": [1.4409212026632645, 0.22186627829371924],
        "corr": 0,
    },
    "LYA_8": {
        "kind": "d_m_d_h",
        "z_eff": 2.85,
        "mean": [42.2969414173007, 6.892078233829452],
        "sigma": [1.8592062161450857, 0.26121579311890053],
        "corr": 0,
    },
    "LYA_9": {
        "kind": "d_m_d_h",
        "z_eff": 2.95,
        "mean": [42.973524532000766, 6.642066453593322],
        "sigma": [2.144758858821369, 0.2763557061035959],
        "corr": 0,
    },
    "LYA_10": {
        "kind": "d_m_d_h",
        "z_eff": 3.25,
        "mean": [44.86302604841359, 5.973738720721969],
        "sigma": [1.8274145029903703, 0.18980106303297226],
        "corr": 0,
    },
}


class DESIExtendedLikelihood(GaussianLikelihood):
    """
    DESI Extended Likelihood
    """

    # users can override this if they want to use a different name
    # which can be useful if you want to keep the different likelihoods
    # separately.
    like_name = "desi_bao"
    x_section = "distances"
    x_name = "z"
    y_section = "distances"

    def __init__(self, options):
        data_sets = options.get_string("desi_data_sets")
        data_sets = data_sets.split(",")

        allowed = list(DESI_EXTENDED_DATA_SETS.keys())
        for data_set in data_sets:
            data_set = data_set.strip()
            if data_set not in allowed:
                raise ValueError(
                    f"Unknown DESI-extended data set {data_set}. Valid options are: {allowed} (comma-separated to use more than one)"
                )
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
                # measurements are at the same redshift we only
                # need to store the redshift once, and this should
                # hopefully trigger an error if we mess up later.
                z.append(-1.0)

        kinds = np.array(kinds)
        z = np.array(z)
        mu = np.array(mu)

        # record the indices of the d_v and d_m_d_h measurements
        # for later
        self.dv_index = np.where(kinds == KIND_DV)[0]
        self.dm_index = np.where(kinds == KIND_DM)[0]
        self.dh_index = np.where(kinds == KIND_DH)[0]

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
                C[i, i] = ds["sigma"] ** 2
                i += 1
            else:
                C[i, i] = ds["sigma"][0] ** 2
                C[i + 1, i + 1] = ds["sigma"][1] ** 2
                C[i, i + 1] = C[i + 1, i] = ds["corr"] * ds["sigma"][0] * ds["sigma"][1]
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
            f = scipy.interpolate.interp1d(z_theory, d_v / r_s, kind=self.kind)
            y[self.dv_index] = f(z_data)

        if self.any_dmdh:
            z_data = self.data_x[self.dm_index]

            d_m = block[self.y_section, "d_m"]
            f = scipy.interpolate.interp1d(z_theory, d_m / r_s, kind=self.kind)
            y[self.dm_index] = f(z_data)

            d_h = 1.0 / block[self.y_section, "h"]
            f = scipy.interpolate.interp1d(z_theory, d_h / r_s, kind=self.kind)
            y[self.dh_index] = f(z_data)
        return y


setup, execute, cleanup = DESIExtendedLikelihood.build_module()
