import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

def problem_generator(problem, qubits, layers, chi, qubits_lab=1):
    """
    This function generates everything needed for solving the problem
    INPUT: 
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose among
            ['circle', 'diamond', 'wavy lines']
        -qubits: number of qubits, must be an integer
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account

    OUTPUT:
        -theta: set of parameters needed for the circuit. It is an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. It is an array with shape (qubits, layers, dimension of data)
        -weight: set of parameters needed fot the circuit only if chi == 'weighted_fidelity_chi'. It is an array with shape (classes, qubits)
        -reprs: variable encoding the label states of the different classes
    """
    chi = chi.lower()
    if chi in ['fidelity', 'weighted_fidelity']: chi += '_chi'
    if chi not in ['fidelity_chi', 'weighted_fidelity_chi']:
        raise ValueError('Figure of merit is not valid')
        
    if chi == 'weighted_fidelity_chi' and qubits_lab != 1: 
        qubits_lab = 1
        print('WARNING: number of qubits for the label states has been changed to 1')
    
    problem = problem.lower()
    if problem == 'circle':
        theta, alpha, reprs = q_circle(qubits, layers, qubits_lab, chi)    
    elif problem == 'diamond':
        theta, alpha, reprs = q_diamond(qubits, layers, qubits_lab, chi) 
    elif problem == 'wavy lines':
        theta, alpha, reprs = q_wavy_lines(qubits, layers, qubits_lab, chi)
    elif problem in ['iris', 'iris_ovr']:
        theta, alpha, reprs = q_iris(qubits, layers, qubits_lab, chi)    
    else:
        raise ValueError('Problem is not valid')
        
    if chi == 'fidelity_chi':
        return theta, alpha, reprs
    elif chi == 'weighted_fidelity_chi':
        weights = np.ones((len(reprs), qubits))
        return theta, alpha, weights, reprs
    

def q_circle(qubits, layers, qubits_lab, chi):
    classes = 2
    reprs = representatives(classes, qubits_lab)
    theta = np.random.rand(qubits, layers, 3)
    alpha = np.random.rand(qubits, layers, 2)
    return theta, alpha, reprs


def q_diamond(qubits, layers, qubits_lab, chi):
    classes = 2
    reprs = representatives(classes, qubits_lab)
    theta = np.random.rand(qubits, layers, 3)
    alpha = np.random.rand(qubits, layers, 2)
    return theta, alpha, reprs


def q_wavy_lines(qubits, layers, qubits_lab, chi):
    classes = 4
    reprs = representatives(classes, qubits_lab)
    theta = np.random.rand(qubits, layers, 3)
    alpha = np.random.rand(qubits, layers, 2)
    return theta, alpha, reprs

def q_iris(qubits, layers, qubits_lab, chi):
    classes = 2
    reprs = representatives(classes, qubits_lab)
    theta = np.random.rand(qubits, layers, 3)
    alpha = np.random.rand(qubits, layers, 2)
    return theta, alpha, reprs


def representatives(n_classes, qubits_lab):

    reprs = np.zeros((n_classes, 2**qubits_lab), dtype = 'complex')
    if qubits_lab == 1:
        if n_classes == 0:
            raise ValueError('Nonsense classifier')
        if n_classes == 1:
            raise ValueError('Nonsense classifier')
        if n_classes == 2:
            reprs[0] = np.array([1, 0])
            reprs[1] = np.array([0, 1])
        if n_classes == 3:
            reprs[0] = np.array([1, 0])
            reprs[1] = np.array([1 / 2, np.sqrt(3) / 2])
            reprs[2] = np.array([1 / 2, -np.sqrt(3) / 2])
        if n_classes == 4:
            reprs[0] = np.array([1, 0])
            reprs[1] = np.array([1 / np.sqrt(3), np.sqrt(2 / 3)])
            reprs[2] = np.array([1 / np.sqrt(3), np.exp(1j * 2 * np.pi / 3) * np.sqrt(2 / 3)])
            reprs[3] = np.array([1 / np.sqrt(3), np.exp(-1j * 2 * np.pi / 3) * np.sqrt(2 / 3)])
        if n_classes == 6:
            reprs[0] = np.array([1, 0])
            reprs[1] = np.array([0, 1])
            reprs[2] = 1 / np.sqrt(2) * np.array([1, 1])
            reprs[3] = 1 / np.sqrt(2) * np.array([1, -1])
            reprs[4] = 1 / np.sqrt(2) * np.array([1, 1j])
            reprs[5] = 1 / np.sqrt(2) * np.array([1, -1j])
        
    if qubits_lab == 2:
        if n_classes == 0:
            raise ValueError('Nonsense classifier')
        if n_classes == 1:
            raise ValueError('Nonsense classifier')
        if n_classes == 2:
            reprs[0] = np.array([1, 0, 0, 0])
            reprs[1] = np.array([0, 0, 0, 1])
        if n_classes == 3:
            reprs[0] = np.array([1, 0, 0, 0])
            reprs[1] = np.array([0, 1, 0, 0])
            reprs[2] = np.array([0, 0, 1, 0])
        if n_classes == 4:
            reprs[0] = np.array([1, 0, 0, 0])
            reprs[1] = np.array([0, 1, 0, 0])
            reprs[2] = np.array([0, 0, 1, 0])
            reprs[3] = np.array([0, 0, 0, 1])
        if n_classes > 2**qubits_lab:
            raise ValueError('Too many classes for the given number of label qubits')
    return reprs

def circuit(theta_aux, entanglement):
    """
    This creates the Quantum circuit for the problem using QuantumState.QCircuit
    INPUT: 
        -theta_aux: set of parameters needed for the circuit. It is an array with shape (qubits, layers, 3 or 6). Alpha and x are coded inside theta_aux
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
    OUTPUT:
        -quantum circuits coding the problem and our Ansätze
    """
    hypar = theta_aux.shape #[qubits, layers, params_per_layer]
    entanglement = entanglement.lower()[0]
    if hypar[-1] not in [3, 6]: 
        raise ValueError('The number of parameters per gate is not correct')
    
    if hypar[-1] == 3:
        num_qubits = hypar[0]
        if num_qubits == 1:
            return single_qubit_circuit(theta_aux)
        elif num_qubits == 2 and entanglement == 'n':
            return _qcircuit_2qubit_noentanglement(theta_aux)
        elif num_qubits == 2 and entanglement == 'y':
            return _qcircuit_2qubit_entanglement(theta_aux)
        elif num_qubits == 3 and entanglement == 'n':
            return _qcircuit_3qubit_noentanglement(theta_aux)
        elif num_qubits == 3 and entanglement == 'y':
            return _qcircuit_3qubit_entanglement(theta_aux)
        else:
            raise ValueError('Not Valid')

def single_qubit_circuit(theta_aux):

    qc = QuantumCircuit(1)

    for i in range(theta_aux.shape[1]):
        mat = U3(theta_aux[0, i, :])
        qc.append(UnitaryGate(mat),[0])
        
    return qc

def _qcircuit_2qubit_noentanglement(theta_aux):
    num_qubits = theta_aux.shape[0]
    num_layers = theta_aux.shape[1]
    
    qc = QuantumCircuit(num_qubits)
    
    for l in range(num_layers):
        for q in range(num_qubits):
            # Generate the custom matrix for the current parameters
            mat = U3(theta_aux[q, l, :])
            # Wrap in UnitaryGate and append
            gate = UnitaryGate(mat, label="U3_custom")
            qc.append(gate, [q])
            
    return qc

def _qcircuit_2qubit_entanglement(theta_aux):
    qc = QuantumCircuit(2)
    for l in range(theta_aux.shape[1] - 1):
        for q in range(theta_aux.shape[0]):
            mat = U3(theta_aux[q,l,:])
            qc.append(UnitaryGate(mat), [q])
        qc.cz(0,1)
    for q in range(theta_aux.shape[0]):
        mat = U3(theta_aux[q,-1,:])
        qc.append(UnitaryGate(mat), [q])
    return qc

def _qcircuit_3qubit_noentanglement(theta_aux):
    qc = QuantumCircuit(3)
    for l in range(theta_aux.shape[1]):
        for q in range(3):
            mat = U3(theta_aux[q, l, :])
            gate = UnitaryGate(mat, label="U3_custom")
            qc.append(gate, [q])
    return qc

def _qcircuit_3qubit_entanglement(theta_aux):
    qc = QuantumCircuit(3)
    for l in range(theta_aux.shape[1] - 1):
        for q in range(theta_aux.shape[0]):
            mat = U3(theta_aux[q, l, :])
            qc.append(UnitaryGate(mat), [q])        
        if l%2 == 0:
            qc.cz(0,1)
            qc.cz(1,2)
        elif l%2 == 1:
            qc.cz(0,2)
    for q in range(theta_aux.shape[0]):
        mat=U3(theta_aux[q,-1,:])
        qc.append(UnitaryGate(mat), [q])
    return qc

def U3(theta3):
        t, phi, lam = theta3
        c = np.cos(t / 2)
        s = np.sin(t / 2)
        e_phi = np.exp(1j * phi / 2)
        e_lambda = np.exp(1j * lam / 2)
        
        return np.array([
        [ c * e_phi * e_lambda,        -s * e_phi * np.conj(e_lambda)],
        [ s * np.conj(e_phi) * e_lambda,  c * np.conj(e_phi) * np.conj(e_lambda)]
        ])


def code_coords(theta, alpha, x):  #Encoding of coordinates
    """
    This functions converts theta, alpha and x in a new set of variables encoding the three of them properly
    INPUT:
        -theta: initial point for the theta parameters. The shape must be correct (qubits, layers, 3)
        -alpha: initial point for the alpha parameters. The shape must be correct (qubits, layers, dim)
        -x: one data for training, only the coordinates
    OUTPUT:
        -theta_aux: shifted thetas encoding alpha and x inside. Same shape as theta
    """
    theta_aux = theta.copy()
    qubits = theta.shape[0]
    layers = theta.shape[1]
    for q in range(qubits):
        for l in range(layers):
            if len(x) <= 3:
                for i in range(len(x)):
                    theta_aux[q, l, i] += alpha[q, l, i] * x[i]
            elif len(x) == 4:
                theta_aux[q, l, 0] += alpha[q, l, 0] * x[0]
                theta_aux[q, l, 1] += alpha[q, l, 1] * x[1]
                theta_aux[q, l, 3] += alpha[q, l, 2] * x[2]
                theta_aux[q, l, 4] += alpha[q, l, 3] * x[3]
            else:
                raise ValueError('Data has too many dimensions')
    
    return theta_aux
