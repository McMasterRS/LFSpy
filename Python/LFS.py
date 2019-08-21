# class stub for interfacing with scikit learn pipeline.
# Fit, transform, and fit_transform methods are required

# gamma:   impurity level (default: 0.2)
# tau:     number of iterations (default: 2)
# sigma:   controls neighboring samples weighting (default: 1)
# alpha:   maximum number of selected feature for each representative point
# NBeta:   numer of distinct \beta (default: 20)
# NRRP:    number of iterations for randomized rounding process (defaul 2000)

# 11 with kiret

import numpy as np
from scipy.io import loadmat
from scipy.optimize import linprog

from fn.fes import fes
from fn.evaluation import evaluation
from fn.classification import classification
from fn.accuracy import accuracy

WANDERING = 0


###########################################
# Local Feature Selection
###########################################

def linprog_callback(res):
    pass
    
class LocalFeatureSelection():

    def __init__(self, alpha=19, gamma=0.2, tau=2, sigma=1, n_beta=20, nrrp=2000, knn=1):
        self.alpha = alpha         # , maximum number of selected features for each representative point
        self.gamma = gamma         # , impurity level (default: 0.2)
        self.tau = tau             # , number of iterations (default: 2)
        self.sigma = sigma         # , controls neighboring samples weighting (default: 1)
        self.n_beta = n_beta       # , number of distinct beta (default: 20)
        self.nrrp = nrrp           # number of iterations for randomized rounding process (default: 2000)
        self.knn = knn             # Possibly k nearest neighbours?

    # (160 ways to describe this thing, 100 times we observed it, and two ways it can be (0 or 1)) 
    def fit(self, training_data, training_labels):

        self.fstar = np.zeros(training_data.shape)          # selected features for each representative point; if fstar(i, j) = 1, ith feature is selected for jth representative point.
        self.fstar_lin = np.zeros(training_data.shape)      # fstar before applying randomized rounding process
        m_features, n_observations = training_data.shape    # (M, N) M number of candidate features, N observations (160x100)
        n_total_cls = [
            np.sum(training_labels^1),                 # the total number of class 0 we observed
            np.sum(training_labels)                    # the total number of class 1 we observed
        ]

        random_numbers = loadmat('./data/r')['r']
        tb_temp = np.zeros((m_features, n_observations, self.n_beta))
        tr_temp = np.zeros((m_features, n_observations, self.n_beta))

        for i in range(self.tau):

            feasibility = np.zeros((n_observations, self.n_beta)) # Whether this was feasible of not
            radious = np.zeros((n_observations, self.n_beta)) # radious is the distance needed from each point, to find something
            b_ratio = np.zeros((n_observations, self.n_beta)) # B ratio?

            n_remaining_cls = [0, 0]

            # Calculate fstar for each observation
            mask_excluded = np.ones(n_observations, dtype=bool)
            without_observation = np.ones(n_observations, dtype=bool)

            # I forget to clean something up, so its wrong answers for every run but the first
            for j in range(0, n_observations): # temporarily set to 2, change back to 0 in production
                mask_excluded[j] = False # Mask out the j'th observation
                
                data_observation = training_data[:, j][..., None] # data for this observation
                matching_class = training_labels[0, j] # class for this observation
                nonmatching_class = matching_class^1 # Binary XOR inversion of the active class (0->1, 1->0)

                labels_excluded = training_labels[0, mask_excluded].astype(bool) # this is all labels excluding the j'th
                data_excluded = training_data[:, mask_excluded]

                n_remaining_cls[matching_class] = n_total_cls[matching_class] - 1 # How many of the same class did we see, excluding this observation
                n_remaining_cls[nonmatching_class] = n_total_cls[nonmatching_class]

                # how similar is this feature from all other features? delta between 
                theta = (-data_excluded[:, ~labels_excluded] + data_observation)**2    # How close is this observation from all other observations of class 0?
                ro = (-data_excluded[:, labels_excluded] + data_observation)**2        # How close is this observation from all other observations of class 1?

                observation_delta = [
                    (-data_excluded[:, ~labels_excluded] + training_data[:, j][..., None])**2, # (theta) How similar is this observation to all other observations of class 0
                    (-data_excluded[:, labels_excluded] + training_data[:, j][..., None])**2   # (ro)   How similar is this observation to all other observations of class 1
                ]

                weight_matrix = np.zeros((n_observations, n_observations)) # distance matrix for i'th feature
                # calcualt
                for k in range(n_observations):
                    dist_ro = np.sqrt(np.sum(self.fstar[:, j][..., None]*observation_delta[1], axis=0))
                    dist_alpha = np.sqrt(np.sum(self.fstar[:, j][..., None]*observation_delta[0], axis=0))
                    weight_matrix[k, mask_excluded] = np.concatenate(( 
                        np.exp(-dist_ro - np.min(dist_ro) ** 2 / self.alpha),
                        np.exp(-dist_alpha - np.min(dist_alpha) ** 2 / self.alpha)
                    )) 

                # what is the average difference.
                average_weight = np.mean(weight_matrix, axis=0)[mask_excluded]
                w1 = average_weight[:n_remaining_cls[1]] # I think this is the average weight for all class 1
                w2 = average_weight[n_remaining_cls[1]:] # i think this is the average weight for all class 2
                normalized_w1 = w1/np.sum(w1)
                normalized_w2 = w2/np.sum(w2)
                # apply random wandering
                w1 = (1-WANDERING) * normalized_w1 + WANDERING * np.random.rand(1, w1.shape[0]) # Oh shit, hidden rand! I think with random wandering at 0, this just equals normalized weights
                w2 = (1-WANDERING) * normalized_w2 + WANDERING * np.random.rand(1, w2.shape[0]) # Oh shit, hidden rand!
                # renormalize
                normalized_w1 = w1/np.sum(w1)
                normalized_w2 = w2/np.sum(w2)
                f_sparsity = 0

                # so for each observation this is the sum of all the difference between it and all other observations
                f_cls = [
                    np.sum(normalized_w2 * theta, axis=1)[..., None],
                    np.sum(normalized_w1 * ro, axis=1)[..., None]
                ]

                # in some cases this leans one way or the other, we want to find the smallest delta, we are finding the minimum  
                A_ub_0 = np.concatenate((np.ones((1, m_features)), -np.ones((1, m_features))), axis=0) # The inequality constraint matrix. Each row of A_ub specifies the coefficients of a linear inequality constraint on x.
                b_ub_0 = np.array([[self.alpha], [-1]]) # The inequality constraint vector. Each element represents an upper bound on the corresponding value of A_ub @ x.

                a = f_cls[matching_class].T / n_remaining_cls[matching_class]
                b = (f_sparsity + f_cls[nonmatching_class]).T / n_remaining_cls[nonmatching_class] # Coefficients of the linear objective function to be minimized.                
                linprog_res_0 = linprog(-b.T, A_ub=A_ub_0, b_ub=b_ub_0, bounds=(0, 1), method='interior-point', options={'tol':0.000001, 'maxiter': 200})

                x = linprog_res_0.x
                fun = linprog_res_0.fun
                slack = linprog_res_0.slack
                con = linprog_res_0.con
                success = linprog_res_0.success
                status = linprog_res_0.status
                nit = linprog_res_0.nit
                message = linprog_res_0.message


                if linprog_res_0.success:
                    epsilon_max = -linprog_res_0.fun

                mask_excluded[j] = True # reset the mask

                for k in range(0, self.n_beta):
                    print("-", k)
                    beta = 1/self.n_beta * (k+1)
                    epsilon = beta * epsilon_max
                    A_ub_1 = np.vstack((np.ones((1, m_features)), -np.ones((1, m_features)), -b)) # BB TODO: Rename
                    b_ub_1 = np.vstack((self.alpha, -1, -epsilon)) # b1 TODO: Rename
                    # NOTE: These share names with the other ones, rename to be distinct
                    

                    linprog_res_1 = linprog(a.T, A_ub=A_ub_1, b_ub=b_ub_1, callback=linprog_callback, bounds=(0.0, 1.0), method='interior-point', options={ 'tol':1e-6, 'maxiter': 200})

                    x = linprog_res_1.x
                    fun = linprog_res_1.fun
                    slack = linprog_res_1.slack
                    con = linprog_res_1.con
                    success = linprog_res_1.success
                    status = linprog_res_1.status
                    nit = linprog_res_1.nit
                    message = linprog_res_1.message

                    if linprog_res_1.success:
                        A = random_numbers <= linprog_res_1.x[..., None]
                        unq_observations = np.unique(A.T, axis=0).T # we only care about unique observations, that is if two are the same, we dont want to iterate over them
                        n_unq_observations = unq_observations.shape[1]
                        feasibility_temp = np.zeros((1, n_unq_observations))
                        radious_temp = np.zeros((1, n_unq_observations))
                        distance_within = np.inf * np.ones((1, n_unq_observations))
                        # distance_between = -np.inf * np.ones((1, n_unq_observations))
                        dr = np.zeros((1, n_unq_observations))
                        far = np.zeros((1, n_unq_observations))

                        for l in range(n_unq_observations):
                            unq_observations_slice = unq_observations[:, l]
                            rep_points_p = training_data[unq_observations_slice, :]
                            if np.sum(A_ub_1 @ unq_observations_slice > b_ub_1[:, 0]) == 0:  # if atleast one feature is active and no more than maxNoFea [b_ub_1[:, 0] is alpha] 
                                feasibility_temp[0, l] = 1
                                distance_within [0, l] = a @ unq_observations_slice
                                # distance_between [0, l] = b @ unq_observations_slice
                                if np.sum(unq_observations_slice^1) < m_features:
                                    dist_rep = np.abs(np.sqrt(np.sum((-rep_points_p + rep_points_p[:, j][..., None]) ** 2, 0)))
                                    unq_dist_rep = np.msort(np.unique(dist_rep)) # a sorted list of all unique distances from representative points
                                    n_cls_closest = []
                                    n_cls_count = [0, 0]
                                    count_m = 0
                                    distance_found = False

                                    # Increase the distance until we have more dissimilar observations than similar observations
                                    while not distance_found:
                                        within_zone = (dist_rep <= unq_dist_rep[count_m]) # which observations are at this distance?
                                        n_cls_closest = [
                                            np.sum((training_labels^1)[0, within_zone]), # the number of 0's within the zone
                                            np.sum(training_labels[0, within_zone]) # the number of 1's within the zone
                                        ]
                                        n_cls_closest[matching_class] = n_cls_closest[matching_class] - 1 # we subtract 1 since we arnt including this point
                                        # check if there are more dissimilar features than similar features close by (proportional to the total number of each)
                                        # this is relative proportions to the original data set
                                        if self.gamma * (n_cls_closest[matching_class]/(n_total_cls[matching_class]-1)) < (n_cls_closest[nonmatching_class]/n_total_cls[nonmatching_class]):
                                            distance_found = True
                                            if count_m == 0:
                                                radious_temp[0, l] = unq_dist_rep[0] # this is the radious of how far we need to go for this
                                            else:
                                                radious_temp[0, l] = 0.5 * (unq_dist_rep[count_m-1] + unq_dist_rep[count_m]) # this is the radious of how far we need to go for this
                                                n_cls_count[matching_class] = n_cls_count[matching_class] + 1
                                            if radious_temp[0, l] == 0: # if the radious is less 0, that is if the point is right on top of it, then pad it a bit
                                                radious_temp[0, l] = 0.000001

                                            within_zone = (dist_rep <= radious_temp[0, l]) # how many are within the zone
                                            rep_points_p_slice = rep_points_p[:, (within_zone == 1)] # these are which points are within the zone
                                            rep_points_p_slice_labels = training_labels[0, (within_zone == 1)]
                                            dr[0, l] = 0
                                            far[0, l] = 0
        
                                            for u in range(rep_points_p_slice.shape[1]):
                                                dist_quasi_test = np.absolute(np.sqrt(np.sum((rep_points_p - rep_points_p_slice[:, u][...,None]) ** 2, axis=0))) # distance from this point to all other points
                                                dist_quasi_test_cls = rep_points_p_slice_labels[u]
                                                min_uniq = np.sort(np.unique(dist_quasi_test))
                                                count_n = 0
                                                total_nearest_neighbours = 0
                                                #  Searches until it finds k nearest neighbours
                                                while total_nearest_neighbours < self.knn+1:
                                                    nearest_neighbours = dist_quasi_test <= min_uniq[count_n] # from smallest to largest, tries to find k nearest neighbours
                                                    total_nearest_neighbours = np.sum(nearest_neighbours)
                                                    count_n = count_n + 1
                                                n_nearest_neighbours = [np.sum(nearest_neighbours & (training_labels^1)), np.sum(nearest_neighbours & training_labels)] # number of nearest neighbours
                                                # n_nearest_neighbours[dist_quasi_test_cls] = n_nearest_neighbours[dist_quasi_test_cls] - 1
                                                # if there are more nearest neighbours of the class that quasi_test is, then increase dr
                                                # if dist_quasi_test_cls == 0 and n_nearest_neighbours[1] < (n_nearest_neighbours[0] - 1):
                                                #     dr[0, l] = dr[0, l] + 1
                                                # # if there are more nearest neighbours of the class that quasi_test is, then increase dr
                                                # elif dist_quasi_test_cls == 1 and (n_nearest_neighbours[1]-1) > n_nearest_neighbours[0]:
                                                #     far[0, l] = far[0, l] + 1

                                                No_NN_C1 = n_nearest_neighbours[1]
                                                No_NN_C2 = n_nearest_neighbours[0]
                                                if matching_class==1:
                                                    if dist_quasi_test_cls==1 and (No_NN_C1-1)>No_NN_C2:
                                                        dr[0, l] = dr[0, l] + 1
                                                    if dist_quasi_test_cls==0 and No_NN_C1>(No_NN_C2-1):
                                                        far[0, l] = far[0, l] + 1

                                                if matching_class==0:
                                                    if dist_quasi_test_cls==1 and (No_NN_C1-1)<No_NN_C2:
                                                        far[0, l] = far[0, l] + 1
                                                    if dist_quasi_test_cls==0 and No_NN_C1<(No_NN_C2-1):
                                                        dr[0, l] = dr[0, l] + 1
                                        count_m = count_m + 1
                        eval_criteria = [
                            dr/n_total_cls[0]-far/n_total_cls[1],
                            dr/n_total_cls[1]-far/n_total_cls[0]
                        ]
                        i_lowest_distance_within = np.argmin(distance_within) # find the shortest within distance
                        TT_binary = unq_observations[:, i_lowest_distance_within]
                        feasibility[j, k] = feasibility_temp[0, i_lowest_distance_within]
                        radious[j, k] = radious_temp[0, i_lowest_distance_within]
                        b_ratio[j, k] = eval_criteria[training_labels[0, j]][0, i_lowest_distance_within]

                        if feasibility[j, k] == 1:
                            tb_temp[:, j, k] = TT_binary
                            tr_temp[:, j, k] = linprog_res_1.x
            b_ratio[feasibility == 0] = -np.inf
            I1 = np.argmax(b_ratio, axis=1) # what column (observation) contains the largest value for each row (feature)
            for j in range(n_observations):
                self.fstar[:, j] = tb_temp[:, i, I1[j]] # what class is associated with the observation that has the largest value for this feature?
                self.fstar_lin[:, j] = tr_temp[:, i, I1[j]]
        print(self.fstar)

    def transform(self, training_data, training_labels, testing_data, testing_labels):
        self.classification_error = [None, None],       # classification error (in percent) associated to the input test points with class 0 and 1
        self.total_classification_error = None          # total error (in percent) for the entire input test points.

    def fit_transform(self, training_data, training_labels, testing_data, testing_labels):
        pass

    def run(self, training_data, training_labels, testing_data, testing_labels):

        training_labels = training_labels.astype(bool)

        self.training_data = training_data                  # (M, N) M is number of candidate fetatures, N is an observation; 
        self.testing_data = testing_data                    # (M, K) M is number of candidate features; K is number of test points.

        self.training_labels = training_labels.astype(bool) # class label = {0,1}
        self.testing_labels = testing_labels                # class label = {0, 1}

        self.results = {
            'fstar': np.zeros(training_data.shape),     # selected features for each representative point; if fstar(i, j)=1, ith feature is selected for jth representative point.
            'fstar_lin': np.zeros(training_data.shape), # fstar before applying randomized rounding process
            'classification_error': [None, None],       # classification error (in percent) associated to the input test points with class 0 and 1
            'total_classification_error': None          # total error (in percent) for the entire input test points.
        }

        M, N = training_data.shape
        fstar = np.zeros((M, N))
        fstar_lin = np.zeros((M, N))

        for i in range(self.tau):
            a, b, epsilon_max = fes(training_data, training_labels, fstar, params)
            t_binary_temp, t_r_temp, b_ratio, feasibility, radious = evaluation(training_data, training_labels, a, b, epsilon_max, params)
            b_ratio[feasibility == 0] = -np.inf
            I1 = np.argmax(b_ratio, axis=1)
            for j in range(N):
                fstar[:, i] = t_binary_temp[:, i, I1[i]]
                fstar_lin[:, i] = t_r_temp[:, i, I1[i]]

        s_class_1, s_class_2 = classification(training_data, training_labels, N, testing_data, fstar, params) # DONE
        classification_error_0, classification_error_1, total_classification_error = accuracy(s_class_1, s_class_2, test_labels) # DONE

        self.fstar = fstar
        self.fstar_lin = fstar_lin
        self.classification_error = [classification_error_0, classification_error_1]
        self.total_classification_error = total_classification_error

###########################################

mat = loadmat('./data/matlab_Data')

training_data = mat['Train']
training_labels = mat['TrainLables']
testing_data = mat['Test']
testing_labels = mat['TestLables']

lfs = LocalFeatureSelection()
lfs.fit(training_data, training_labels)
print(lfs.fstar)

# if targets(1,y)==1
# if targets(1,Q(u))==1 && (No_NN_C1-1)>No_NN_C2
# if targets(1,Q(u))==0 && No_NN_C1>(No_NN_C2-1)

# if targets(1,y)==0
# quasiTest_Class=targets(1,Q(u));
# if quasiTest_Class==1 && (No_NN_C1-1)<No_NN_C2
# if quasiTest_Class==0 && No_NN_C1<(No_NN_C2-1)

