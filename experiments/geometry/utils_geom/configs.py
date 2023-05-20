import cv2

# Fundamental matrix.
BASE_FRANSAC_CFG = {
    "method": cv2.USAC_DEFAULT,  # LO-RANSAC
    "maxIters": 10_000,  # (default = 1_000)
    "ransacReprojThreshold": 1.0,  # (default = 3.0)
    "confidence": 0.999,  # (default=0.99)
}


# Essential estimation is not considered since USAC does not work in
# Windows (LAPACK is not bundled with pip).
# BASE_ERANSAC_CFG = {
#     "method": cv2.USAC_DEFAULT, # LO-RANSAC
#     "maxIters": 1000, # (default=1000). This parameter is hardcoded in OpenCV (bug).
#     "threshold": 1.0, # (default=1.0)
#     "prob": 0.999, # (default=0.999)
#     }


# P3P ransac to estimate the inliers from tentative 3d-2d correspondences.
BASE_USAC_P3P = cv2.UsacParams()
BASE_USAC_P3P.confidence = 0.999  # default=0.99
BASE_USAC_P3P.maxIterations = 10_000  # default=5_000
# sample used for LO refinement.
BASE_USAC_P3P.loSampleSize = 10  # hardcoded maximum = 15
BASE_USAC_P3P.loIterations = 4  # hardcoded maximum = 15
# DLS is not iterative refinement (otherwise LOCAL_OPTIM_INNER_AND_ITER_LO)
BASE_USAC_P3P.loMethod = cv2.LOCAL_OPTIM_INNER_LO
BASE_USAC_P3P.sampler = cv2.SAMPLING_UNIFORM
BASE_USAC_P3P.score = cv2.SCORE_METHOD_MSAC
BASE_USAC_P3P.threshold = 1.0  # default=1.5


def calibrate_thresholds(K, *args):
    """ In-place calibration of thresholds to correspond w. normalized coords.

    Args:
        K: (3, 3) calibration matrix of the form
            [[fx, 0, cx]
             [0, fy, cy]
             [0, 0,  1]]
        args: sequence of dictionaries/UsacParams isntances.
    """
    fx, fy = K[0, 0], K[1, 1]
    factor = 1.0 / (0.5 * (fx + fy))

    for cfg in args:
        if isinstance(cfg, dict):

            key = [k for k in cfg.keys() if 'threshold' in k.lower()]
            assert len(key) == 1

            cfg[key[0]] *= factor

        elif isinstance(cfg, cv2.UsacParams):
            cfg.threshold *= factor

        else:
            raise TypeError
