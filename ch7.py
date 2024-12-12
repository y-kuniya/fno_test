# Poisson 1dのデータを生成
#11/27
# Poisson eqのinput関数を生成する関数を作成
#11/30
# Poisson eqのデータを保存

import numpy as np
import scipy
from scipy.special import eval_chebyt
from scipy import sparse
from scipy.sparse.linalg import spsolve

def make_differential_ops(nx,dx):
    f0 = np.identity(nx+1)
    f1 = np.roll(f0,1,axis=1)
    f_1= f1.transpose()
    deriv2 = (f1+f_1-2.0*f0)/(dx**2)
    return sparse.csr_matrix(deriv2[1:-1,1:-1])

def random_func(x,random_coeff):
    f_x = 0.0
    for i in range(random_coeff.size):
        f_x  += random_coeff[i]*eval_chebyt(i,x)
    return f_x 

def make_training_dataset(num_data=10,num_x=50,dim_chebyt=10):
    eval_x = np.linspace(0,1,num_x+1)
    lmat = make_differential_ops(num_x,1/num_x)

    input_list = []
    target_list= []
    for i in range(num_data):
        func_coeff = np.random.randn(dim_chebyt)
        input_func = random_func(eval_x,func_coeff)
        input_data = np.vstack([eval_x,input_func])
        solved_func= spsolve(lmat,input_func[1:-1])
        solved_func= np.concatenate([[0],solved_func,[0]]) #境界条件の追加

        input_list.append(input_data)
        target_list.append(solved_func)

    input_list = np.array(input_list)
    target_list= np.array(target_list)
    np.savez("./poisson_1d_data/data.npz",input=input_list,target=target_list)
    
def main():
    make_training_dataset()

if __name__ == '__main__':
    main()
