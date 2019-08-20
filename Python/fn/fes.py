import numpy as np
from scipy.optimize import linprog

# sigma: Controls neighboring samples weighting (default: 1)
# alpha: Maximum number of selected feature for each representative point (default: 19)
def fes(self, training_data, training_labels, fstar):

    sigma = self.sigma
    alpha = self.alpha

    WANDERING = 0

    M, N = training_data.shape

    # the total number of each class
    n_total_cls = [np.sum(~training_labels), np.sum(training_labels)]

    a = np.zeros((N, M))
    b = np.zeros((N, M))
    epsilon_max = np.zeros((N, 1))

    mask = np.ones(N, dtype=bool)

    # For each training observation, runs something N x N times
    for i in range(N):
        mask[i] = False

        selected_class = training_labels[0, i] # This is the class of the currently selected feature
        temp_training = training_data[:, mask] # training data for all but ith column
        temp_labels = training_labels[:, mask].astype(bool) # training data for all but ith column
        n_excluded_cls = [0, 0]
        n_excluded_cls[selected_class] = n_total_cls[selected_class] - 1
        n_excluded_cls[~selected_class] = n_total_cls[~selected_class]

        ro = (-temp_training[:, temp_labels[0]] + training_data[:, i][..., None])**2
        theta = (-temp_training[:, ~temp_labels[0]] + training_data[:, i][..., None])**2

        w = np.zeros((N, N))
        w_ave = np.zeros((1,N))

        # For each training observation
        for j in range(N):
            fstar_slice = fstar[:, j][..., None]
            dist_ro = np.sqrt(np.sum(fstar_slice*ro, axis=0))
            dist_alpha = np.sqrt(np.sum(fstar_slice*theta, axis=0))
            w[j, mask] = np.concatenate(( 
                np.exp(-dist_ro - np.min(dist_ro) ** 2 / alpha),
                np.exp(-dist_alpha - np.min(dist_alpha) ** 2 / alpha)
            ))
        
        w_ave = np.mean(w, axis=0)[mask]
        w1 = w_ave[:n_excluded_cls[1]]
        w2 = w_ave[n_excluded_cls[1]:]
        normalized_w1 = w1/np.sum(w1)
        normalized_w2 = w2/np.sum(w2)
        w1 = (1-WANDERING) * normalized_w1 + WANDERING * np.random.rand(1, w1.shape[0])
        w2 = (1-WANDERING) * normalized_w2 + WANDERING * np.random.rand(1, w2.shape[0])   
        normalized_w1 = w1/np.sum(w1)
        normalized_w2 = w2/np.sum(w2)
        w = np.concatenate((normalized_w1.T, normalized_w2.T, np.zeros((1, 1))), axis=0) # we append the two and add a zero for padding
        f_sparsity = w[-1] * np.ones((M, 1))

        f_cls_0_temp = np.zeros((M, 1))
        f_cls_1_temp = np.zeros((M, 1))

        for j in range(n_excluded_cls[0]): # sum up all of the somethings
            f_cls_0_temp = f_cls_0_temp + (w[j + n_excluded_cls[1]] * theta[:, j])[..., None]
        for j in range(n_excluded_cls[1]): # sum up all of the somethings
            f_cls_1_temp = f_cls_1_temp + (w[j] * ro[:, j])[..., None]

        if selected_class == 0:
            b[i, :] = (f_sparsity + f_cls_1_temp).T / n_excluded_cls[1]
            a[i, :] = f_cls_0_temp.T / n_excluded_cls[0]
        elif selected_class == 1:
            b[i, :] = (f_sparsity + f_cls_0_temp).T / n_excluded_cls[0]
            a[i, :] = f_cls_1_temp.T / n_excluded_cls[1]

        A_ub = np.concatenate((np.ones((1, M)), -np.ones((1, M))), axis=0)
        b_ub = np.array([[alpha, -1]]).T

        res = linprog(-b[i, :].T, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method='interior-point', options={'tol':0.000001, 'maxiter': 200})

        if res.success:
            epsilon_max[i] = -res.fun

        mask[i] = True

    return [a, b, epsilon_max]