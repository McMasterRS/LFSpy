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

random_numbers = loadmat('./data/r')['r']

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
        pass

    def transform(self, training_data, training_labels, testing_data, testing_labels):
        self.classification_error = [None, None],       # classification error (in percent) associated to the input test points with class 0 and 1
        self.total_classification_error = None          # total error (in percent) for the entire input test points.

    def fit_transform(self, training_data, training_labels, testing_data, testing_labels):
        pass

    # FES might be feature estimation!
    def fit(self, training_data, training_labels, test_data, test_labels):

        self.fstar = np.zeros(training_data.shape)          # selected features for each representative point; if fstar(i, j) = 1, ith feature is selected for jth representative point.
        self.fstar_lin = np.zeros(training_data.shape)      # fstar before applying randomized rounding process

        m_features, n_observations = training_data.shape    # (M, N) M number of candidate features, N observations (160x100)
        n_total_cls = [np.sum(training_labels^1), np.sum(training_labels)] # Total number of each class in our training data
        training_labels = training_labels[0] # I want this as a 1 dimensional array so I can use it as a mask throughout

        overall_feasibility = np.zeros((n_observations, self.n_beta)) # Whether this was feasible of not
        overall_radious = np.zeros((n_observations, self.n_beta)) # radious is the distance needed from each point, to find something
        overall_b_ratio = np.zeros((n_observations, self.n_beta)) # B ratio?
        tb_temp = np.zeros((m_features, n_observations, self.n_beta))
        tr_temp = np.zeros((m_features, n_observations, self.n_beta))

        # For each observation, we calculate an fstar value for each feature
        # for i_observation, selected_observation in enumerate(training_data.T, start=0):
        for z in range(self.tau):
            for i_observation in range(0, n_observations): # Leave one out class testing

                selected_observation = training_data[:, i_observation][..., None]
                selected_label = training_labels[i_observation]
                matching_label = selected_label
                nonmatching_label = selected_label^1

                not_selected_obs = np.ones((1, n_observations), dtype=bool)[0] # mask for all observations not at i
                not_selected_obs[i_observation] = False # deselect the i'th observations
                excluding_selected = not_selected_obs

                excluded_observations = training_data[:, not_selected_obs] # get all observations not at i
                excluded_labels = training_labels[not_selected_obs] # get the labels not at i, (convert to bool so we can use it as a mask)
                n_excluded_class = [np.sum(excluded_labels^1), np.sum(excluded_labels)]

                # basically, if we already suspect the class for each feature from previous iterations, we bump it in that direction
                fstar_mask = self.fstar[:, i_observation].astype(bool) # fstar is binary if a feature is relevent for this observations

                # Calculate the difference between this obseration and all other observations
                observation_deltas = (training_data - selected_observation)**2
                # Possibly logistic regression weights?
                # Calculate a weight for each observation based on if we've previously found fstar values for its features
                fstar_observation_deltas = (np.sqrt(np.sum(observation_deltas[fstar_mask, :], axis=0))) # total similarity along features selected by fstar
                observation_weight = np.zeros((n_observations, n_observations))
                observation_weight[:, training_labels == 0] = np.exp((-(fstar_observation_deltas[training_labels == 0] - np.min(fstar_observation_deltas[training_labels == 0]))**2)/self.sigma)
                observation_weight[:, training_labels == 1] = np.exp((-(fstar_observation_deltas[training_labels == 1] - np.min(fstar_observation_deltas[training_labels == 1]))**2)/self.sigma)
                observation_weight[:, i_observation] = 0
                average_observation_weight = np.mean(observation_weight, axis=0) # mean weight of all observations
                normalized_weight = np.zeros((1, n_observations)) # normalized weight for all observations not including the current observation
                normalized_weight[:, training_labels == 0] = np.round(average_observation_weight[training_labels == 0]/np.sum(average_observation_weight[training_labels == 0]), decimals=16)
                normalized_weight[:, training_labels == 1] = np.round(average_observation_weight[training_labels == 1]/np.sum(average_observation_weight[training_labels == 1]), decimals=16)

                # Find the average weighted difference for each feature
                average_feature_deltas = [0, 0]
                average_feature_deltas[matching_label] = np.round(np.sum(normalized_weight[0, excluding_selected & (training_labels == matching_label)] * observation_deltas[:, excluding_selected & (training_labels == matching_label)], axis=1) / (n_total_cls[matching_label]-1), decimals=16)
                average_feature_deltas[nonmatching_label] = np.sum(normalized_weight[0, excluding_selected & (training_labels == nonmatching_label)] * observation_deltas[:, excluding_selected & (training_labels == nonmatching_label)], axis=1)/n_total_cls[nonmatching_label]

                # 
                A_ub_0 = np.concatenate((np.ones((1, m_features)), -np.ones((1, m_features))), axis=0) # The inequality constraint matrix. Each row of A_ub specifies the coefficients of a linear inequality constraint on x.
                b_ub_0 = np.array([[self.alpha], [-1]]) # The inequality constraint vector. Each element represents an upper bound on the corresponding value of A_ub @ x.

                linprog_res_0 = linprog(-average_feature_deltas[nonmatching_label], A_ub=A_ub_0, b_ub=b_ub_0, bounds=(0, 1), method='interior-point', options={'tol':0.000001, 'maxiter': 200})             # This is secretly a maximization function

                if linprog_res_0.success:
                    epsilon_max = -linprog_res_0.fun

                    for i_beta in range(0, self.n_beta): # beta is kind of the granularity or resolution, higher = better estimation?
                        beta = np.round(1/self.n_beta * (i_beta + 1), decimals=15)
                        epsilon = beta * epsilon_max

                        A_ub_1 = np.vstack((np.ones((1, m_features)), -np.ones((1, m_features)), -average_feature_deltas[nonmatching_label])) # BB TODO: Rename
                        b_ub_1 = np.vstack((self.alpha, -1, -epsilon)) # b1 TODO: Rename

                        # linprog_res_1.x is our best estimate for which class is associated with what feature.
                        linprog_res_1 = linprog(average_feature_deltas[matching_label], A_ub=A_ub_1, b_ub=b_ub_1, callback=linprog_callback, bounds=(0.0, 1.0), method='interior-point', options={ 'tol':1e-6, 'maxiter': 200})

                        class_estimations = linprog_res_1.x[..., None]

    ###################################################################################################
    ###################################################################################################

                        if linprog_res_1.success:
                            # Random rounding, for each of our estimates that are close to 0.5 (were in the middle of what class it should be)
                            # basically this tries to find where we could use more accuracy, then tries all options, and revaluates to see if we did better
                            
                            requires_adjustment = random_numbers <= class_estimations # compare our class estimations against random numbers, where our results are close to 0.5 we will get true, otherwise False
                            unique_options = np.unique(requires_adjustment, axis=1) # this will result in all probable options for what our less certain features can be
                            n_options = unique_options.shape[1]

                            option_feasabilities = np.zeros((1, n_options))[0] # not all options are feasible, this is adjusted as feasible options are found
                            option_radiuses = np.zeros((1, n_options))[0]
                            option_delta_within = np.inf * np.ones((1, n_options))[0]

                            dr = np.zeros((1, n_options))
                            far = np.zeros((1, n_options))

                            # Each option is a probable case for which features could be further classified. We try each option, and find the one that best fits
                            for i_option, option in enumerate(unique_options.T): # For each probable option
                                if np.sum(A_ub_1 @ option > b_ub_1[:, 0]) == 0:  # if atleast one feature is active and no more than maxNoFeatures
                                    if np.sum(option) > 0: # If there is atleast one relevant feature in this option

                                        option_feasabilities[i_option] = 1 # this is a feasible option if the above criteria is fulfilled
                                        representative_points = training_data[option, :] # Each feature that has been selected in this option
                                        option_delta_within[i_option] = average_feature_deltas[matching_label] @ option  # get our previously calculated feature delta for the selected features
                                        active_point = representative_points[:, i_observation][..., None]
                                        rep_deltas = np.abs(np.sqrt(np.sum((-representative_points + active_point) ** 2, 0))) # We get the difference between this active point and all other rep points
                                        unique_deltas = np.msort(np.unique(rep_deltas)) # we filter out all duplicate deltas and sort in ascending order in order to find the smallest delta

                                        # Increase the difference threshold until we have more dissimilar observations than similar observations
                                        for i_delta, delta in enumerate(unique_deltas, start=0):

                                            radious = delta
                                            observations_within_delta = (rep_deltas <= delta) # find all representative points that are atleast delta different
                                            n_cls_within = [
                                                np.sum((training_labels^1)[observations_within_delta]),   # the number of 0's that fall within this difference threshold
                                                np.sum(training_labels[observations_within_delta])        # the number of 1's within the zone
                                            ]
                                            n_cls_within[selected_label] = n_cls_within[selected_label] - 1     # we subtract 1 since we arnt including our selected point
                                            # We want there to be less of our similar class proportionally at this delta than our dissimilar class
                                            if self.gamma * (n_cls_within[selected_label]/(n_total_cls[selected_label]-1)) < (n_cls_within[selected_label^1]/n_total_cls[selected_label^1]):
                                                if i_delta > 0:
                                                    radious = 0.5 * (radious + unique_deltas[i_delta-1]) # this is the radious of how far we need to go for this
                                                    n_cls_within[selected_label] = n_cls_within[selected_label] + 1 # increment the cound of how many classes are within this radious

                                                if radious == 0: # if the radious is 0, that is probably if the point is right on top of it, then pad it a bit
                                                    radious = 0.000001

                                                observations_within_radious = (rep_deltas <= radious) # how many are within the zone
                                                rep_points_within_radious = representative_points[:, (observations_within_radious == 1)] # these are which points are within the zone
                                                classes_within_radious = training_labels[observations_within_radious == 1]
                                                dr[0, i_option] = 0
                                                far[0, i_option] = 0

                                                for i_point, rep_point_within in enumerate(rep_points_within_radious.T):
                                                    rep_point_within = rep_point_within[..., None]
                                                    dist_quasi_test = np.absolute(np.sqrt(np.sum((representative_points - rep_point_within) ** 2, axis=0))) # delta between this point and all other points
                                                    dist_quasi_test_cls = classes_within_radious[i_point]
                                                    min_uniq = np.sort(np.unique(dist_quasi_test)) # we once again sort the features by ascending difference
                                                    total_nearest_neighbours = 0
                                                    #  Searches until it finds k nearest neighbours
                                                    for i_delta_within, delta_within in enumerate(min_uniq):
                                                        nearest_neighbours = dist_quasi_test <= min_uniq[i_delta_within] # from smallest to largest, tries to find k nearest neighbours
                                                        total_nearest_neighbours = np.sum(nearest_neighbours)
                                                        if(total_nearest_neighbours > self.knn):
                                                            break

                                                    n_nearest_neighbours = [  # number of nearest neighbours of each class
                                                        np.sum(nearest_neighbours & (training_labels^1)),
                                                        np.sum(nearest_neighbours & training_labels)
                                                    ]

                                                    # The case where the this point's class is in the majority amongst neighbouring points in the localized radious
                                                    if dist_quasi_test_cls == selected_label and (n_nearest_neighbours[selected_label] - 1) > n_nearest_neighbours[selected_label^1]:
                                                        dr[0, i_option] = dr[0, i_option] + 1 # Count the number of points that are in the majority

                                                    # The case where the this point's class is in the minority amongst neighbouring points in the localized radious, that is there are more dissimilar points within the radious
                                                    if dist_quasi_test_cls == (selected_label^1) and n_nearest_neighbours[selected_label] > (n_nearest_neighbours[selected_label^1]-1):
                                                        far[0, i_option] = far[0, i_option] +1 # count the number of times points are in the minority
                                                break

                            eval_criteria = [
                                dr/n_total_cls[0] - far/n_total_cls[1], # we get the difference between the proportion of similar to dissimilar neighbouring points within the radious
                                dr/n_total_cls[1] - far/n_total_cls[0]
                            ]
                            i_lowest_distance_within = np.argmin(option_delta_within) # find the shortest within distance
                            TT_binary = unique_options[:, i_lowest_distance_within]
                            overall_feasibility[i_observation, i_beta] = option_feasabilities[i_lowest_distance_within]
                            overall_radious[i_observation, i_beta] = option_radiuses[i_lowest_distance_within]
                            overall_b_ratio[i_observation, i_beta] = eval_criteria[training_labels[i_observation]][0, i_lowest_distance_within]
                            # if overall_b_ratio[i_observation, i_beta] != _bratio[i_observation, i_beta]:
                            #     for i in range(160):
                            #         print(_TT[i_observation, i_beta, i], class_estimations[i, 0])
                            #     print('oops')
                            if overall_feasibility[i_observation, i_beta] == 1:
                                tb_temp[:, i_observation, i_beta] = TT_binary
                                tr_temp[:, i_observation, i_beta] = linprog_res_1.x

                bratio = overall_b_ratio
                overall_b_ratio[overall_feasibility == 0] = -np.inf
                I1 = np.argmax(overall_b_ratio, axis=1) # what column (observation) contains the largest value for each row (feature)

                for j in range(n_observations):
                    self.fstar[:, j] = tb_temp[:, i_observation, I1[j]] # what class is associated with the observation that has the largest value for this feature?
                    self.fstar_lin[:, j] = tr_temp[:, i_observation, I1[j]]
            print('done')

        SClass1,SClass2 = self.classification(training_data, training_labels, 100, test_data, self.fstar, self.gamma, self.knn)
        ErCls1, ErCls2, ErClassification = self.accuracy(SClass1, SClass2, test_labels)
        print(ErCls1, ErCls2, ErClassification)

    def classification(self, patterns, targets, N, test, fstar, gamma, knn):
        """ """

        H = test.shape[1]
        s_class_1_sphere_knn = np.zeros((1, H))
        s_class_2_sphere_knn = np.zeros((1, H))

        for t in range(H) :
            s_class_1_sphere_knn[0, t], s_class_2_sphere_knn[0, t], _ = self.class_sim_m(test[:, t], N, patterns, targets, fstar, gamma, knn)

        return [s_class_1_sphere_knn, s_class_2_sphere_knn]

    def class_sim_m(self, test, N, patterns, targets, fstar, gamma, knn):
        
        targets = targets[None, ...]

        n_nt_cls_l = np.sum(targets) # count the number of targets
        n_nt_cls_2 = N-n_nt_cls_l # fidn the remaining somethings
        M = patterns.shape[0] 
        NC1 = 0
        NC2 = 0
        S = np.Inf * np.ones((1, N))

        NoNNC1knn = np.zeros((1,N))
        NoNNC2knn = np.zeros((1,N))
        NoNNC1 = np.zeros((1,N))
        NoNNC2 = np.zeros((1,N))
        radious = np.zeros((1,N))
        for i in range(N) :
            
            XpatternsPr = patterns*fstar[:, i][...,None]
            testPr = test*fstar[:, i]
            Dist = np.abs(np.sqrt(np.sum((-testPr[...,None] + XpatternsPr)**2,0)))
            # Dist = np.abs(np.sqrt(np.sum((XpatternsPr-testPr[..., None])**2,1)))
            min1 = np.msort(Dist)
            
            min_Uniq = np.unique(min1)
            m = -1
            No_nereser = 0
            while No_nereser<knn:
                m = m+1
                a1 = min_Uniq[m]
                NN = Dist <= a1
                No_nereser = np.sum(NN)
            NoNNC1knn[0, i] = np.sum(NN & targets)
            NoNNC2knn[0, i] = np.sum(NN & ~targets)
            
            A = np.where(fstar[:,i] == 0)
            if A[0].shape[0] <M:
                a_mask = np.ones(patterns.shape[0], dtype=bool)
                a_mask[A] = False
                patterns_P = patterns[a_mask]
                test_P = test[a_mask]
                # test_P[A,:] = []
                testA = patterns_P[:,i] - test_P
                Dist_test = np.abs(
                    np.sqrt(
                        np.sum(
                            (patterns_P[:,i]-test_P)**2
                        ,0)
                    )
                )
                Dist_pat = np.abs(
                    np.sqrt(
                        np.sum(
                            (patterns_P-patterns_P[:, i][..., None])**2
                        ,0)
                    )
                )
                EE_Rep = np.msort(Dist_pat)
                remove = 0
                if targets[0, i] == 1:
                    UNQ = np.unique(EE_Rep)
                    k = -1
                    NC1 = NC1+1
                    if remove !=  1:
                        Next = 1
                        while Next == 1:
                            k = k+1
                            r = UNQ[k]
                            F1 = (Dist_pat == r)
                            NoCls1r = np.sum(F1 & targets)
                            NoCls2r = np.sum(F1 & ~targets)
                            F2 = (Dist_pat <= r)
                            NoCls1clst = np.sum(F2 & targets)-1
                            NoCls2clst = np.sum(F2 & ~targets)
                            if gamma*(NoCls1clst/(n_nt_cls_l-1))<(NoCls2clst/n_nt_cls_2):
                                Next = 0
                                if (k-1) == 0:
                                    r = UNQ[k]
                                else:
                                    r = 0.5*(UNQ[k-1]+UNQ[k])
                                
                                if r == 0:
                                    r = 10**-6
                                
                                r = 1*r
                                F2 = (Dist_pat <= r)
                                NoCls1clst = np.sum(F2 & targets)-1
                                NoCls2clst = np.sum(F2 & ~targets)
                        if Dist_test <= r:
                            patterns_P = patterns*fstar[:,i][...,None]
                            test_P = test*fstar[:,i]
                            Dist = np.abs(
                                np.sqrt(
                                    np.sum(
                                        (patterns_P-test_P[...,None])**2,0
                                        )))
                            min1 = np.msort(Dist)
                            min_Uniq = np.unique(min1)
                            m = -1
                            No_nereser = 0
                            while No_nereser<knn:
                                m = m+1
                                a1 = min_Uniq[m]
                                NN = Dist <= a1
                                No_nereser = np.sum(NN)
                            
                            NoNNC1[0,i] = np.sum(NN & targets)
                            NoNNC2[0,i] = np.sum(NN & ~targets)
                            if NoNNC1[0,i]>NoNNC2[0,i]:
                                S[0,i] = 1
                                
                if targets[0,i] == 0:
                    UNQ = np.unique(EE_Rep)
                    k = -1
                    NC2 = NC2+1
                    if remove !=  1:
                        Next = 1
                        while Next == 1:
                            k = k+1
                            r = UNQ[k]
                            F1 = (Dist_pat == r)
                            NoCls1r = np.sum(F1 & targets)
                            NoCls2r = np.sum(F1 & ~targets)
                            F2 = (Dist_pat <= r)
                            NoCls1clst = np.sum(F2 & targets)
                            NoCls2clst = np.sum(F2 & ~targets)-1
                            if  gamma*(NoCls2clst/(n_nt_cls_2-1))<(NoCls1clst/n_nt_cls_l):
                                Next = 0
                                if (k-1) == 0:
                                    r = UNQ[k]
                                else:
                                    r = 0.5*(UNQ[k-1]+UNQ[k])
                                
                                if r == 0:
                                    r = 10**-6
                                
                                r = 1*r
                                F2 = (Dist_pat <= r)
                                NoCls1clst = np.sum(F2 & targets)
                                NoCls2clst = np.sum(F2 & ~targets)-1
                        
                        if Dist_test <= r:
                            patterns_P = patterns*fstar[:,i][..., None]
                            test_P = test*fstar[:, i]
                            Dist = np.abs(
                                np.sqrt(
                                    np.sum(
                                        (patterns_P-test_P[..., None])**2,0)))
                            min1 = np.msort(Dist)
                            min_Uniq = np.unique(min1)
                            m = -1
                            No_nereser = 0
                            while No_nereser<knn:
                                m = m+1
                                a1 = min_Uniq[m]
                                NN = Dist <= a1
                                No_nereser = np.sum(NN)
                        
                            NoNNC1[0,i] = np.sum(NN & targets)
                            NoNNC2[0,i] = np.sum(NN & ~targets)
                            if NoNNC2[0,i] > NoNNC1[0,i]:
                                S[0,i] = 1
            radious[0, i] = r

        Q1 = (NoNNC1) > (NoNNC2knn)
        Q2 = (NoNNC2) > (NoNNC1knn)
        S_Class1 = np.sum(Q1 & targets) / NC1
        S_Class2 = np.sum(Q2 & ~targets) / NC2

        if S_Class1 == 0 and S_Class2 == 0:
            Q1 = (NoNNC1knn) > (NoNNC2knn)
            Q2 = (NoNNC2knn) > (NoNNC1knn)
            S_Class1 = np.sum(Q1 & targets)/NC1
            test = Q2 & np.logical_not(targets)
            S_Class2 = np.sum(Q2 & np.logical_not(targets))/NC2

        return [S_Class1, S_Class2, radious]

    def accuracy(self, s_class_1, s_class_2, test_tables):

        n_test = test_tables.shape[1]
        er_classification = np.zeros((1, 1))
        er_cl_s_2 = np.zeros((1, 1))
        er_cl_s_1 = np.zeros((1, 1))
        dc_ls_1 = np.zeros((1, 1))
        dc_ls_2 = np.zeros((1, 1))
        n_cls_1_test = np.sum(test_tables)
        n_cls_2_test = n_test - n_cls_1_test
        n = 0

        DQ0 = test_tables == 0
        SS1 = s_class_1
        SS2 = s_class_2
        DQ1 = (SS1) > (SS2)
        DQ1 = np.double(DQ1)
        DQ1[DQ0] = 0
        dc_ls_1[n] = np.sum(np.sum(DQ1))
        er_cl_s_1[n] = 1 - dc_ls_1[n] / n_cls_1_test
        er_cl_s_1[n] = er_cl_s_1[n] * 100

        DQ0_SCZ = test_tables == 0
        SS1_SCZ = s_class_1
        SS2_SCZ = s_class_2
        DQ1_SCZ = (SS1_SCZ) < (SS2_SCZ)
        DQ1_SCZ = np.double(DQ1_SCZ)
        DQ1_SCZ[~DQ0_SCZ] = 0
        dc_ls_2[n] = np.sum(np.sum(DQ1_SCZ))
        er_cl_s_2[n] = 1-dc_ls_2[n] / n_cls_2_test
        er_cl_s_2[n] = er_cl_s_2[n] * 100

        er_classification[n] = 1 - (dc_ls_1[n] + dc_ls_2[n]) / n_test
        er_classification[n] = er_classification[n] * 100

        return er_cl_s_1, er_cl_s_2, er_classification

mat = loadmat('./data/matlab_Data')

training_data = mat['Train']
training_labels = mat['TrainLables']
testing_data = mat['Test']
testing_labels = mat['TestLables']

data = loadmat('./data/matlab')

# fstar = data['fstar']
# gamma=0.2
# knn=1


lfs = LocalFeatureSelection()
lfs.fit(training_data, training_labels, testing_data, testing_labels)
# SClass1,SClass2 = lfs.classification(training_data, training_labels, training_labels.shape[1], testing_data, fstar, gamma, knn)
# er_cl_s_1, er_cl_s_2, er_classification = lfs.accuracy(SClass1,SClass2, testing_labels)