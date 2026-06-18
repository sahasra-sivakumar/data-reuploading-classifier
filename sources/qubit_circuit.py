import numpy as np
import matplotlib.pyplot as plt
import re

from datasets import *
from problem_gen import *
from test_set import *
from save_data import *
from fid_minimization import *
from weight_fid_minimization import *

problems = ['circle', 'diamond', 'wavy lines', 'iris']


def minimizer(chi, problem, qubits, entanglement, layers, method, name, iris_id,
              seed = 30, epochs=3000, batch_size=20,  eta=0.1):
    """
    This function creates data and minimizes whichever problem (from the selected ones) 
    INPUT:
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose among
            ['circle', 'diamond', 'wavy lines', 'iris']
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method - only L-BFGS-B was used
        -name: a name we want for our our files to be save with
        -iris_id: to use the iris dataset
        -seed: seed of numpy.random, needed for replicating results
    OUTPUT:
        This function has got no outputs, but several files are saved in an appropiate folder. The files are
        -summary.txt: Saves useful information for the problem
        -theta.txt: saves the theta parameters as a flat array
        -alpha.txt: saves the alpha parameters as a flat array
        -weight.txt: saves the weights as a flat array if they exist
    """
    np.random.seed(seed)

    if problem == 'iris':
        (train_data, test_data), drawing = data_generator(problem, iris_id)
    else:
        data,drawing = data_generator(problem, iris_id=None)
        train_data = data[:20]# if changed needs to be changed in save_data.py line 145 too
        test_data = data[20:]

    print(len(train_data)+ len(test_data))
    #train_data = data[:20]# if changed needs to be changed in save_data.py line 145 too
    #test_data = data[20:]
   
    if chi == 'fidelity_chi':
        qubits_lab = qubits
        theta, alpha, reprs = problem_generator(problem,qubits, layers, chi,
                                            qubits_lab=qubits_lab)
        theta, alpha, f, loss_hist = fidelity_minimization(theta, alpha, train_data, reprs,
                                            entanglement, method, 
                                            batch_size, eta, epochs)
        acc_train = tester(theta, alpha, train_data, reprs, entanglement, chi)
        acc_test = tester(theta, alpha, test_data, reprs, entanglement, chi)
        write_summary(chi, problem, qubits, entanglement, layers, method, name, iris_id,
              theta, alpha, 0, loss_hist, f, acc_train, acc_test, seed, epochs=epochs)
    elif chi == 'weighted_fidelity_chi':
        qubits_lab = 1
        theta, alpha, weight, reprs = problem_generator(problem,qubits, layers, chi,
                                            qubits_lab=qubits_lab)
        theta, alpha, weight, f, loss_hist = weighted_fidelity_minimization(theta, alpha, weight, train_data, reprs,
                                            entanglement, method)
        acc_train = tester(theta, alpha, train_data, reprs, entanglement, chi, weights=weight)
        acc_test = tester(theta, alpha, test_data, reprs, entanglement, chi, weights=weight)
        write_summary(chi, problem, qubits, entanglement, layers, method, name, iris_id,
              theta, alpha, weight, loss_hist, f, acc_train, acc_test, seed, epochs=epochs)
        
#============LOSS FUNCTION PLOT =====================
def plot_loss_tot(chi, problem, qubits, entanglement, layers, method):
    plt.figure(figsize = (7,7))

    for i in layers:
        name_fold = name_folder(chi, problem, qubits, entanglement, i, method)
        file_text = np.loadtxt(name_fold + '/' + 'run_loss.txt')
        iteration = file_text[:,0]
        cost = file_text[:,1]
        if chi == 'fidelity_chi':
            plt.plot(iteration, 1+cost, marker = 'o', linewidth = 1, markersize = 2, linestyle='-', label = str(i)+' layer')
        else:
            plt.plot(iteration, cost, marker = 'o', linewidth = 1, markersize = 2, linestyle='-', label = str(i)+' layer')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.grid()
    plt.legend()
    plt.show()

#==========ACCURACY PLOT=====================
def plot_acc_tot(chi, problem, qubits, entanglement, layers, method):

    layers_list, acc_trains, acc_tests = [], [], []

    for i in layers:
        name_fold = name_folder(chi, problem, qubits, entanglement, i, method)
        summary = os.path.join(name_fold, "run_summary.txt")

        if not os.path.exists(summary):
            print(f"  [missing] {summary}")
            
        acc_train = acc_test = None
        with open(summary) as f:
            for line in f:
                m = re.match(r"acc_train\s*=\s*([0-9.]+)", line.strip())
                if m:
                    acc_train = float(m.group(1))
                m = re.match(r"acc_test\s*=\s*([0-9.]+)", line.strip())
                if m:
                    acc_test = float(m.group(1))

        if acc_train is not None and acc_test is not None:
            layers_list.append(i)
            acc_trains.append(acc_train)
            acc_tests.append(acc_test)
            print(f"  layers={i:2d}  train={acc_train:.3f}  test={acc_test:.3f}")
        else:
            print(f"  [parse error] {summary}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers_list, acc_trains, "o-",  label="acc_train", linewidth=2)
    ax.plot(layers_list, acc_tests,  "s--", label="acc_test",  linewidth=2)

    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{chi} | {problem} | {qubits} qubit(s) | entanglement = {entanglement} | {method}")
    ax.set_xticks(layers_list)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

                
