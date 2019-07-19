import numpy as np
import ClassSimM as ClassSimM

def classification(patterns,targets,N,test,fstar,gamma,knn):

    H=shape[test,2]
    S_Class1_sphereKNN=np.zeros[1,H]
    S_Class2_sphereKNN=np.zeros[1,H]
    for t in range(H) :
        [S_Class1_sphereKNN[1,t], S_Class2_sphereKNN[1,t]]= ClassSimM(test[:,t],N,patterns,targets,fstar,gamma,knn)
            

    return [S_Class1_sphereKNN,S_Class2_sphereKNN]
