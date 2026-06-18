from scipy.optimize import minimize
import numpy as np
from qiskit.quantum_info import Statevector, state_fidelity

from problem_gen import *


def fidelity_minimization(theta, alpha, train_data, reprs, 
                       entanglement, method,
                       batch_size, eta, epochs):
    """
    This function takes the parameters of a problem and computes the optimal parameters for it, using different functions. It uses the fidelity minimization
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -train_data: set of data for training. There must be several entries (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -method: minimization method, to choose among ['SGD', another valid for function scipy.optimize.minimize]
        -batch_size: size of the batches for stochastic gradient descent, only for 'SGD' method
        -eta: learning rate, only for 'SGD' method
        -epochs: number of epochs , only for 'SGD' method
    OUTPUT:
        -theta: optimized point for the theta parameters. The shape is correct (qubits, layers, 3)
        -alpha: optimized point for the alpha parameters. The shape is correct (qubits, layers, dim)
        -chi: value of the minimization function
    """
    params, hypars = _translate_to_scipy(theta, alpha)

    iteration = [0]
    loss_history = []

    def callback(params):
        iteration[0] += 1
        cost = _scipy_minimizing(params, hypars, train_data, reprs, entanglement)
        loss_history.append((iteration[0], float(cost)))
        print(f"Iteration {iteration[0]}, cost: {cost:.6f}")

    results = minimize(_scipy_minimizing, params, 
                        args = (hypars, train_data, reprs, entanglement),
                                method=method, callback = callback)
        
    theta, alpha = _translate_from_scipy(results['x'], hypars)
        

    return theta, alpha, results['fun'], loss_history


def _translate_to_scipy(theta, alpha):
    """
    This function is a intermediate step for translating theta and alpha to a single variable for scipy.optimize.minimize
    """
    qubits = theta.shape[0]
    layers = theta.shape[1]
    dim = alpha.shape[-1]
    
    return np.concatenate((theta.flatten(), alpha.flatten())), (qubits, layers, dim)


def _translate_from_scipy(params, hypars):
    """
    This function is a intermediate step for getting theta and alpha from a single variable for scipy.optimize.minimize
    """
    (qubits, layers, dim) = hypars
    if dim <= 3:
        theta = params[:qubits * layers * 3]. reshape(qubits, layers, 3)
        alpha = params[qubits * layers * 3: qubits * layers * 3 + qubits * layers * dim].reshape(qubits, layers, dim)
        
    if dim == 4:
        theta = params[:qubits * layers * 6]. reshape(qubits, layers, 6)
        alpha = params[qubits * layers * 6: qubits * layers * 6 + qubits * layers * dim].reshape(qubits, layers, dim)
    return theta, alpha


def _scipy_minimizing(params, hypars, train_data, reprs, entanglement):
    """
    This function returns the chi^2 function for using scipy
    INPUT:
        -params: theta and alpha inside the same variable
        -hypars: hyperparameters needed to rebuild theta and alpha
        -train_data: training dataset for the classifier
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT:
        - -Av_chi_square, which is the function we want to minimize
    """
    theta, alpha = _translate_from_scipy(params, hypars)
    return -Av_chi_square(theta, alpha, train_data, reprs, entanglement)


def _chi_square(theta, alpha, data, reprs, entanglement): #Chi for one point
    """
    This function compute chi^2 for only one point
    INPUT: 
        -theta: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, dimension of data)
        -data: one data for training. It must be (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT: 
        -chi^2 for data
    """
    #
    x, y = data
    theta_aux = code_coords(theta, alpha, x)
    C = circuit(theta_aux, entanglement)
    psi = Statevector(C)
    ans = np.sqrt(state_fidelity(reprs[y], psi))
    return ans


def Av_chi_square(theta, alpha, train_data, reprs, entanglement): #Chi in average
    """
    This function compute chi^2 for only one point
    INPUT: 
        -theta: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, dimension of data)
        -data: one data for training. It must be (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT: 
        -Averaged chi^2 for data
    """
    Av_Chi = 0
    for d in train_data:
        Av_Chi += _chi_square(theta, alpha, d, reprs, entanglement)

    return Av_Chi / len(train_data)



