from math import sqrt, sin, cos

import numpy as np
from numba import njit


@njit('f8[:,:,::1](f8[:,::1])', cache=True)
def jacobian_SE3action_at0(p):
    """Jacobian of the SE3 action on a set of points p:

        d               │
       ────── exp(xi) p │  ,    xi = [v^T, w^T]^T
        d xi            │0

    Args:
        p: (3, n) array of 3d coordinates.

    Returns:
        J: (n, 3, 6) jacobian
    """
    out = np.zeros((p.shape[1], 3, 6))

    # derivative w.r.t. v (translation part).
    out[:, 0, 0] = 1.0
    out[:, 1, 1] = 1.0
    out[:, 2, 2] = 1.0

    # derivative w.r.t. w (rotation part).
    out[:, 1, 5] = p[0]
    out[:, 2, 4] = -p[0]
    out[:, 0, 5] = -p[1]
    out[:, 2, 3] = p[1]
    out[:, 0, 4] = p[2]
    out[:, 1, 3] = -p[2]

    return out


@njit('Tuple((f8[:,:,::1], f8[:,::1]))(f8[:,::1])', cache=True)
def jacobian_dprojection_dy(y):
    """Jacobian of projection function w.r.t. set of points being projected."""
    y_inv = 1.0 / y[2]
    y_inv_sq = y_inv * y_inv

    J = np.zeros((y.shape[1], 2, 3))
    J[:, 0, 0] = y_inv
    J[:, 1, 1] = y_inv
    J[:, 0, 2] = - y_inv_sq * y[0]
    J[:, 1, 2] = - y_inv_sq * y[1]

    u = y[:2] * y_inv

    return J, u


@njit('Tuple((f8[:,:,::1], f8[:,::1]))(f8[:,::1], f8[:,::1], f8[:,::1],'
      ' f8[:,::1])', fastmath=True, cache=True)
def jacobian_dprojection_dposeSE3(X, K, R, t):
    """ Jacobian of projection function w.r.t. se(3) form of the camera pose:

    Considering the projection function as:
        u = \pi(y) = proj(R X + t),
    the following Jacobian is computed:
        d u    d                                    │
        ──── = ──── proj(Exp(xi)[(R X + t)^T, 1]^T) │
        d xi   d xi                                 │0

    Args:
        X: (3, n) array of 3d coordinates.
        K: (3, 3) calibration matrix.
        R: (3, 3) rotation matrix.
        t: (3, 1) translation vector.

    Returns:
        J: (n, 2, 6) jacobian of each projection "u" w.r.t. each point in X.
        u: (2, n) projections on image plane.
    """
    n = X.shape[1]
    J = np.empty((n, 2, 6))

    X_c = R @ X + t
    J1, u = jacobian_dprojection_dy(K.dot(X_c))
    J2 = jacobian_SE3action_at0(X_c)

    for i in range(n):
        J[i] = J1[i] @ K @ J2[i]
    # J = J @ K @ jacobian_SE3action_at0(X_c)
    return J, u


@njit('Tuple((f8[:,:,::1], f8[:,::1]))(f8[:,::1], f8[:,::1], f8[:,::1],'
      ' f8[:,::1], f8[:,::1])', cache=True)
def jacobians_and_residuals_pose(X, p, K, R, t):
    J, u = jacobian_dprojection_dposeSE3(X, K, R, t)  # (n, 2, 6), (2, n)
    res = u - p  # (2, n)
    return J, res


@njit('Tuple((f8[:,::1], f8[:,::1]))(f8[::1])', cache=True)
def se3_exp_map(twist):
    """SE3 exponential map.

    Args:
        twist: (6,) elemente of se3 in its vectorized form.

    Returns:
        R: (3, 3) rotation matrix
        t: (3, 1) translation vector
    """
    eps = 1e-10

    v = twist[:3]
    w = twist[3:]

    theta_sq = w.dot(w)
    theta = sqrt(theta_sq)

    w_hat = np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])
    w_hat_sq = w_hat @ w_hat

    I_3 = np.eye(3)

    if theta < eps:
        R = I_3 + w_hat + 0.5 * w_hat_sq
        V = I_3 + 0.5 * w_hat + w_hat_sq / 6.0
        t = V.dot(v)
        return R, np.expand_dims(t, axis=1)

    # trigonometri terms.
    s = sin(theta)
    cos_t = (1.0 - cos(theta)) / theta_sq
    sin_t = (theta - s) / (theta_sq * theta)

    R = I_3 + s / theta * w_hat + cos_t * w_hat_sq
    V = I_3 + cos_t * w_hat + sin_t * w_hat_sq
    t = V.dot(v)
    return R, np.expand_dims(t, axis=1)


@njit('Tuple((f8[:,::1], f8[:,::1]))(f8[::1], f8[:,::1], f8[:,::1])', cache=True)
def SE3_left_update(step, R_current, t_current):
    # apply exponential map.
    R_step, t_step = se3_exp_map(step)
    R = R_step.dot(R_current)
    t = R_step.dot(t_current) + t_step
    return R, t


def pose_LM_optimization(
        X, p, K, R, t, sigmas2d=None, sigmas3d=None, niter=5):
    """LM optimization of the pose, by minimizig reprojection errors."""
    eps = 1e-10

    # increase/ decrase factor of influence for the gradient.
    damping_factor_inc = 1e1
    damping_factor_dec = 1.0 / damping_factor_inc

    use_2d_covs = sigmas2d is not None
    use_3d_covs = sigmas3d is not None

    if use_3d_covs:
        assert use_2d_covs

    if use_2d_covs:
        if use_3d_covs:
            KR = K.dot(R)
        else:
            sigmas2d_inv = np.linalg.inv(sigmas2d)

    # LM iterative optimization.
    R_current, t_current = R.copy(), t.copy()

    for i in range(niter):
        # J: (n, 2, 6), res: (2, n)
        J, res = jacobians_and_residuals_pose(X, p, K, R_current, t_current)

        # construct approximate Hessian.
        if use_2d_covs and use_3d_covs:
            # 2d uncertainty + propagated 3d uncertainties.
            Jproj = jacobian_dprojection_dy(KR @ X)[0] @ KR  # (n, 2, 3)
            W = np.linalg.inv(
                sigmas2d + Jproj @ sigmas3d @ Jproj.transpose(0, 2, 1) + eps
            )
            JTW = J.transpose(0, 2, 1) @ W

        elif use_2d_covs:
            JTW = J.transpose(0, 2, 1) @ sigmas2d_inv
        else:
            JTW = J.transpose(0, 2, 1)

        JTWJ = np.einsum('ndi, nie -> de', JTW, J)  # (6, 6)

        # get writeable view of the diagonal.
        diag = np.einsum('ii -> i', JTWJ)

        # define damping term.
        if i == 0:
            # damping initialization based on [Hartley, Zisserman, 2004].
            # 1e-3 * (trace(JTWJ) / d)
            damping = 2e-4 * diag.sum()
            cost_prev = np.einsum('dn, dn -> ', res, res)
        else:
            # update damping based on current cost.
            cost = np.einsum('dn, dn -> ', res, res)
            if cost < cost_prev:
                damping *= damping_factor_dec
            else:
                damping *= damping_factor_inc
            cost_prev = cost

        # augment approximate Hessian (inplace modification).
        diag += damping * diag

        # solve step and update state.
        step = np.linalg.solve(JTWJ, -np.einsum('ndi, in -> d', JTW, res))
        R_current, t_current = SE3_left_update(step, R_current, t_current)

        # if i == niter-1:
        #     # final covariance matrix of the point coordinates.
        #     J, _ = jacobians_and_residuals_pose(
        #         X, p, K, R_current, t_current)

        #     if use_2d_covs and use_3d_covs:
        #         # 2d uncertainty + propagated 3d uncertainties.
        #         Jproj, _ = jacobian_dprojection_dy(KR @ X) @ KR # (n, 2, 3)
        #         W = np.linalg.inv(
        #             sigmas2d + Jproj @ sigmas3d @ Jproj.transpose(0, 2, 1) + eps
        #             )
        #         JTW = J.transpose(0, 2, 1) @ W

        #     elif use_2d_covs:
        #         JTW = J.transpose(0, 2, 1) @ sigmas2d_inv
        #     else:
        #         JTW = J.transpose(0, 2, 1)
        #     JTWJ = np.einsum('ndi, nie -> de', JTW, J) # (6, 6)
        #     sigma_pose = np.linalg.inv(JTWJ)

    return R_current, t_current
    # return R_current, t_current, sigma_pose


if __name__ == '__main__':
    rng = np.random.default_rng(0)

    from scipy.stats import special_ortho_group as SO
    from autograd import jacobian
    import autograd.numpy as anp

    so3_rng = SO(3, seed=rng)
    so2_rng = SO(2, seed=rng)

    def error(delta, X, p, K, R, t):
        Rstep, tstep = se3_exp_map_autograd(delta)

        Rupdt = Rstep @ R
        tupdt = Rstep @ t + tstep

        y = K @ (Rupdt @ X + tupdt)
        u = y[:2] / y[2:3]
        return u - p

    def se3_exp_map_autograd(twist):
        """SE3 exponential map.

        Args:
            twist: (6,) elemente of se3 in its vectorized form.

        Returns:
            R: (3, 3) rotation matrix
            t: (3, 1) translation vector
        """
        eps = 1e-10

        v = twist[:3]
        w = twist[3:]

        theta_sq = anp.dot(w, w)
        theta = anp.sqrt(theta_sq)

        w_hat = anp.array([
            [0.0, -w[2], w[1]],
            [w[2], 0.0, -w[0]],
            [-w[1], w[0], 0.0],
        ])
        w_hat_sq = w_hat @ w_hat

        I_3 = anp.eye(3)

        if theta < eps:
            R = I_3 + w_hat + 0.5 * w_hat_sq
            V = I_3 + 0.5 * w_hat + w_hat_sq / 6.0
            t = anp.dot(V, v)
            return R, anp.expand_dims(t, axis=1)

        # trigonometri terms.
        s = anp.sin(theta)
        cos_t = (1.0 - anp.cos(theta)) / theta_sq
        sin_t = (theta - s) / (theta_sq * theta)

        R = I_3 + s / theta * w_hat + cos_t * w_hat_sq
        V = I_3 + cos_t * w_hat + sin_t * w_hat_sq
        t = anp.dot(V, v)
        return R, anp.expand_dims(t, axis=1)

    def create_2d_covariances(rng, so2_rng, n):
        """Synthetic 2D covariances based on [Vakhitov, 2021]."""
        # setup based on [Vakhitov, 2021].
        ngroups = 10
        n_per_group = n // ngroups
        ntrunc = n % ngroups
        nt = ngroups * n_per_group

        sigma_per_group = np.linspace(1.0, 10.0, ngroups)**2
        sigma1_per_group = rng.uniform(0.0, 1.0, ngroups) * sigma_per_group
        # sigma1_per_group = sigma_per_group  # homogeneous inside group

        covs_per_group = np.zeros((ngroups, 2, 2))
        covs_per_group[:, 0, 0] = sigma_per_group
        covs_per_group[:, 1, 1] = sigma1_per_group

        Rs = so2_rng.rvs(ngroups)
        covs_per_group = Rs @ covs_per_group @ Rs.transpose(0, 2, 1)

        covs = np.repeat(covs_per_group, n_per_group, 0)

        # add truncated covs.
        if ntrunc > 0:
            tcovs = covs[rng.randint(0, nt, ntrunc)]
            if len(tcovs.shape) == 2:
                tcovs = tcovs[None]
            covs = np.concatenate((covs, tcovs), axis=0)

        return covs

    def create_3d_covariances(rng, so3_rng, n):
        """Create synthetic 3D point covariances."""
        # setup based on [Vakhitov, 2021].
        ngroups = 10
        n_per_group = n // ngroups
        ntrunc = n % ngroups
        nt = ngroups * n_per_group

        sigma_per_group = np.linspace(0.01, 0.1, ngroups)**2
        sigma1_per_group = rng.uniform(0.0, 1.0, ngroups) * sigma_per_group
        sigma2_per_group = rng.uniform(0.0, 1.0, ngroups) * sigma_per_group
        # sigma1_per_group = sigma_per_group  # homogeneous inside group
        # sigma2_per_group = sigma_per_group  # homogeneous inside group

        covs_per_group = np.zeros((ngroups, 3, 3))
        covs_per_group[:, 0, 0] = sigma_per_group
        covs_per_group[:, 1, 1] = sigma1_per_group
        covs_per_group[:, 2, 2] = sigma2_per_group

        Rs = so3_rng.rvs(ngroups)
        covs_per_group = Rs @ covs_per_group @ Rs.transpose(0, 2, 1)

        covs = np.repeat(covs_per_group, n_per_group, 0)

        # add truncated covs.
        if ntrunc > 0:
            tcovs = covs[rng.randint(0, nt, ntrunc)]
            if len(tcovs.shape) == 2:
                tcovs = tcovs[None]
            covs = np.concatenate((covs, tcovs), axis=0)

        return covs

    def multivariate_normal_batch(
            rng,
            covs: np.ndarray,
            mu=None,
            nsamples: int = 1) -> np.ndarray:
        """Given (n, d, d) covariances, generate (n, d, nsamples).

        A Gaussian distribution N(mu, covs) is assumed.

        Args:
            rng: numpy random generator.
            covs: (n, d, d) covariance matrices related to a Gaussian random
                variable of dimension d.
            mu: (n, d) means of the random Gaussian variable. If not given, they
                are set to zeros.
            nsamples: number of samples to draw from each distribution.

        Returns:
            noise: (n, d, nsamples)
        """
        n, d, d1 = covs.shape
        assert d == d1, (d, d1)

        if mu is not None:
            if len(mu.shape) == 2:
                mu = mu[:, :, None]
            nm, dm = mu.shape[:2]
            assert nm == n, (nm, n)
            assert dm == d, (dm, d)

        try:
            # sample from standard Gaussian and transform to desired distribution.
            noise = rng.standard_normal(
                d * n * nsamples).reshape(n, d, nsamples)
            noise = np.linalg.cholesky(covs) @ noise
            if mu is not None:
                noise += mu
            noise = noise.squeeze()

        except np.linalg.LinAlgError:
            # Cholesky failed. Check non PD matrices and set zero-noise to them.
            mask_pd = np.linalg.eigvals(covs).min(axis=1) > 0.0
            n_pd = sum(mask_pd)
            noise = np.zeros((n, d, nsamples))

            noise_std = rng.standard_normal(
                d * n_pd * nsamples).reshape(n_pd, d, nsamples)

            noise[mask_pd] = np.linalg.cholesky(covs[mask_pd]) @ noise_std
            if mu is not None:
                noise += mu
            noise = noise.squeeze()

        return noise

    # input.
    R = SO(3).rvs(random_state=rng)
    t = rng.random((3, 1)) * 10

    # calib matrix.
    K = np.array([
        [800., 0., 320.],
        [0., 800., 240.],
        [0., 0., 1.]
    ])
    res = (640, 480)

    # image plane coordinates.
    n = 1000
    p_im = rng.uniform(0., 1., (2, n)) * [[res[0]], [res[1]]]

    # add virtual sensor noise.
    covs2d = create_2d_covariances(rng, so2_rng, n)
    noise_2d = multivariate_normal_batch(rng, covs2d, nsamples=1)

    p_im_noise = p_im + noise_2d.T
    p_im_noise[0] = p_im_noise[0].clip(0, res[0])
    p_im_noise[1] = p_im_noise[1].clip(0, res[1])

    # 3D points in camera reference (without noise, since its ground-truth).
    z_lim = (4., 8.)
    rays = np.linalg.inv(K) @ np.concatenate((p_im, np.ones((1, n))))
    p_c = rng.uniform(z_lim[0], z_lim[1], (1, n)) * rays

    # set translation to the centroid of the random point cloud.
    t_cw = p_c.mean(axis=1, keepdims=True)
    # rotation.
    R_cw = so3_rng.rvs()

    # 3D points in world reference.
    p_w = R_cw.T @ p_c - R_cw.T.dot(t_cw)

    # add virtual noise to 3d points.
    covs3d = create_3d_covariances(rng, so3_rng, n)
    noise_3d = multivariate_normal_batch(rng, covs3d, nsamples=1)
    p_w_noise = p_w + noise_3d.T

    p = p_im_noise
    X = p_w_noise

    # auto.
    delta = anp.zeros((6,))
    jac_fun = jacobian(error)
    jac_auto = jac_fun(delta, X, p, K, R, t)
    # print('\n', jac_auto[:, 0])

    # analytic.
    jac_ana, _ = jacobians_and_residuals_pose(X, p, K, R, t)
    # print('\n', jac_ana[0])

    print('Max diff. analytic vs automatic jacobians:\t'
          f'{anp.max(anp.abs(jac_auto.transpose(1, 0, 2) - jac_ana)):.2e}')
    # print(f'{jac_auto[:, 0]}\n{jac_ana[0]}')

    R_est, t_est = pose_LM_optimization(
        X,
        p,
        K, R_cw, t_cw,
        sigmas2d=covs2d, sigmas3d=covs3d,
        niter=10)

    # errors.
    e_t = np.linalg.norm(t_est - t_cw)

    trace_rot_error_1 = R_cw.ravel().dot(R_est.ravel()) - 1
    e_rot = 180 / np.pi * np.arccos((0.5 * trace_rot_error_1).clip(-1, 1))
    print('with covs:')
    print(f'translation error:\t{e_t:.5f}\nrotation error:\t{e_rot:.5f}\n')

    R_est, t_est = pose_LM_optimization(
        X,
        p,
        K, R_cw, t_cw,
        sigmas2d=None, sigmas3d=None,
        niter=10)

    # errors.
    e_t = np.linalg.norm(t_est - t_cw)

    trace_rot_error_1 = R_cw.ravel().dot(R_est.ravel()) - 1
    e_rot = 180 / np.pi * np.arccos((0.5 * trace_rot_error_1).clip(-1, 1))
    print('without covs:')
    print(f'translation error:\t{e_t:.5f}\nrotation error:\t{e_rot:.5f}\n')
