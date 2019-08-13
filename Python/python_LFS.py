import numpy as np
from FES import FES
from evaluation import evaluation
from classification import classification
from accuracy import accuracy
from scipy.io import loadmat

def LFS(train, train_lables, test, test_labels, para):

    def validate_FES(b, a, EpsilonMax, i):
        data = loadmat('./data/FES'+str(i+1))
        _a = np.isclose(a, data['a'])
        _b = np.isclose(b, data['b'])
        _c = np.isclose(EpsilonMax, data['EpsilonMax'], rtol=0.0001)
        print("validate_FES",_a.all() and _b.all() and _c.all())
        return _a.all() and _b.all() and _c.all()

    impurity_level      = para["gamma"]     # Impurity Level (GAMMA, γ)
    n_iterations        = para["tau"]       # Number of iterations (TAU, T)
    neighbour_weight    = para["alpha"]     # Neighbouring sample weighting (ALPHA, α)
    max_selected_features = para["sigma"]   # Max number of selected features per representative point (SIGMA, σ)
    n_beta              = para["NBeta"]     # number of distinct beta (BETA, β)
    nrrp                = para["NRRP"]      # Number randomized rounding permutations
    m_rows              = train.shape[0]    # Number of rows in the training data
    n_columns           = train.shape[1]    # Number of columns in the training data
    knn                 = 1

    fstar               = np.zeros((m_rows, n_columns))  # Preallocated space for results
    fstar_lin           = np.zeros((m_rows, n_columns))  # preallocated space for results
    
    # On every iteration, compute something
    for j in range(n_iterations):        
        data = loadmat('./data/FES'+str(j+1))

        # compute something probably feature selection?
        b, a, epsilon_max = FES(neighbour_weight, m_rows, n_columns, train, train_lables, fstar, max_selected_features)
        validate_FES(b, a, epsilon_max, j)

        a = data['a']
        b = data['b']
        epsilon_max = data['EpsilonMax']

        # compute something
        tb_temp, tr_temp, b_ratio, feasib, _ = evaluation(neighbour_weight, n_beta, n_columns, epsilon_max, b, a, train, train_lables, impurity_level, nrrp, knn)        
        
        W1 = feasib == 0
        b_ratio[W1] = -np.inf

        ##
        I1 = np.argmax(b_ratio, axis=1) # I want the god damn indices
        for i in range(n_columns):
            fstar[:, i] = tb_temp[ :, i, I1[i] ]
            fstar_lin[:, i] = tr_temp[ :, i, I1[i] ]

    [s_class_1, s_class_2] = classification(train, train_lables, n_columns, test, fstar, impurity_level, knn) # DONE
    [_, _, er_cls_1, er_cls_2, er_classification] = accuracy(s_class_1, s_class_2, test_labels) # DONE
    return [fstar, fstar_lin, er_cls_1, er_cls_2, er_classification]

mat = loadmat('matlab_Data')
i_data = loadmat('interm_data')
comp = loadmat('sample_matlab_results')

fstar, fstar_lin, er_cls_1, er_cls_2, er_classification = LFS(
    mat['Train'],
    mat['TrainLables'],
    mat['Test'],
    mat['TestLables'],{
    'gamma': 0.2,
    'tau': 2,
    'sigma': 1,
    'alpha': 19,
    'NBeta': 20,
    'NRRP': 2000
})

print(fstar, fstar_lin, er_cls_1, er_cls_2, er_classification)
print((fstar == comp['fstar']).all())
print((fstar_lin == comp['fstarLin']).all())
print((er_cls_1 == comp['ErCls1']).all())
print((er_cls_2 == comp['ErCls2']).all())
print((er_classification == comp['ErClassification']).all())