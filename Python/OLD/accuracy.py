# function [dc_ls_2,dc_ls_1,er_cl_s_1,er_cl_s_2,er_classification]=GH_accuracy...
#     (s_class_1,s_class_2,test_tables)


# dc_ls_2, dc_ls_1, er_cl_s_1, er_cl_s_2, er_classification

import numpy as np

def accuracy(s_class_1, s_class_2, test_tables):

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

    return dc_ls_2, dc_ls_1, er_cl_s_1, er_cl_s_2, er_classification