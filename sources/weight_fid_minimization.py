from problem_gen import code_coords, circuit
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace, Statevector
 

def weighted_fidelity_minimization(theta, alpha, weight, train_data, reprs, 
                       entanglement, method):
    """
    This function takes the parameters of a problem and computes the optimal parameters for it, using different functions. It uses the weighted fidelity minimization
    INPUT: 
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -train_data: set of data for training. There must be several entries (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -method: L-BFGS-B
    OUTPUT:
        -theta: optimized point for the theta parameters. The shape is correct (qubits, layers, 3)
        -alpha: optimized point for the alpha parameters. The shape is correct (qubits, layers, dim)
        -chi: value of the minimization function
    """
    
    params, hypars = _translate_to_scipy(theta, alpha, weight)
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
    
    theta, alpha, weight = _translate_from_scipy(results['x'], hypars)
            
    return theta, alpha, weight, results['fun'], loss_history


def mat_fidelities(theta_aux, weight, reprs, entanglement,
                    return_circuit = False):
    """
    This function takes computes fidelities for a given circuit and weigths
    INPUT: 
        -theta_aux: set of parameters needed for the circuit, alpha is encoded here too. It is an array with shape (qubits, layers, 3)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -return_circuit: boolean varaible, True if the circuit is returned
    OUTPUT:
        -Fidelities: relative fidelities for different labels and qubits
        -C: quantum circuit
    """
    labels = weight.shape[0]
    qubits = weight.shape[1]
    Fidelities = np.empty(weight.shape)
    C = circuit(theta_aux, entanglement)
    sv = Statevector(C)
    dm = DensityMatrix(sv)

    for q in range(qubits):
        qubits_to_trace = [i for i in range(qubits) if i != q]
        rdm = partial_trace(dm, qubits_to_trace)
        #rdm = C.reduced_density_matrix(q)
        for l in range(labels):
            Fidelities[l, q] = np.real(rdm.expectation_value(np.outer(reprs[l], np.conj(reprs[l]))))
            
    if return_circuit == False:
        return Fidelities
    
    if return_circuit == True:
        return Fidelities, C

def w_fidelities(Fidelities, weight):
    """
    This function weights fidelities for a given circuit and weigths
    INPUT: 
        -Fidelities: relative fidelities for different labels and qubits
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits)
    OUTPUT:
        -w_fid: weighted fidelities for different labels
    """
    w_fid = np.sum(Fidelities * weight, axis=1)
    return w_fid


def _chi(theta, alpha, weight, d, reprs, entanglement):
    """
    This function compute chi^2 for only one point
    INPUT: 
        -theta: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, dimension of data)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits) 
        -data: one data for training. It must be (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT: 
        -chi^2 for data
    """
    x, y = d
    theta_aux = code_coords(theta, alpha, x)
    fids = mat_fidelities(theta_aux, weight, reprs, entanglement)
    w_fid = w_fidelities(fids, weight)
    if len(w_fid) == 4:
        Y = 1 / 3 * np.ones(len(w_fid))
        Y[y] = 1
    if len(w_fid) == 3:
        Y = 1 / 4 * np.ones(len(w_fid))
        Y[y] = 1
    if len(w_fid) == 2:
        Y = np.zeros(len(w_fid))
        Y[y] = 1
    return 0.5 * np.linalg.norm(w_fid - Y) ** 2


def Av_Chi_Square(theta, alpha, weight, data, reprs, entanglement):
    """
    This function compute chi^2 for only one point
    INPUT: 
        -theta: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, dimension of data)
        -weight: set of parameters needed fot the circuit. Must be an array with shape (classes, qubits) 
        -data: one data for training. It must be (x,y)
        -reprs: variable encoding the label states of the different classes
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT: 
        -Averaged chi^2 for data
    """
    Av_Chi = 0
    for d in data:
        Av_Chi += _chi(theta, alpha, weight, d, reprs, entanglement)
        
    return Av_Chi / len(data)


def _translate_to_scipy(theta, alpha, weight):
    """
    This function is a intermediate step for translating theta and alpha to a single variable for scipy.optimize.minimize
    """
    qubits = theta.shape[0]
    layers = theta.shape[1]
    dim = alpha.shape[-1]
    classes = weight.shape[0]
    
    return np.concatenate((theta.flatten(), alpha.flatten(),weight.flatten())), (qubits, layers, dim, classes)

def _translate_from_scipy(params, hypars):
    """
    This function is a intermediate step for getting theta and alpha from a single variable for scipy.optimize.minimize
    """
    (qubits, layers, dim, classes) = hypars
    if dim <= 3:
        theta = params[:qubits * layers * 3]. reshape(qubits, layers, 3)
        alpha = params[qubits * layers * 3: qubits * layers * 3 + qubits * layers * dim].reshape(qubits, layers, dim)
        weight = params[(qubits * layers * 3 + qubits * layers * dim):].reshape(classes, qubits)
    
    if dim == 4:
        theta = params[:qubits * layers * 6]. reshape(qubits, layers, 6)
        alpha = params[qubits * layers * 6: qubits * layers * 6 + qubits * layers * dim].reshape(qubits, layers, dim)
        weight = params[(qubits * layers * 6 + qubits * layers * dim):].reshape(classes, qubits)
    
    return theta, alpha, weight

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
        - Av_chi_square, which is the function we want to minimize
    """
    theta, alpha, weight = _translate_from_scipy(params, hypars)
    return Av_Chi_Square(theta, alpha, weight, train_data, reprs, entanglement)
