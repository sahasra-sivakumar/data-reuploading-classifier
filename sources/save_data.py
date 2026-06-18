import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap 
from matplotlib.colors import Normalize

from test_set import *
from datasets import *
from problem_gen import *

### save_data ###
def write_summary(chi, problem, qubits, entanglement, layers, method, name, iris_id,
          theta, alpha, weights, loss_history, chi_value, acc_train, acc_test, seed, epochs):
    """
    This function takes some informations of a given problem and saves the text files 
    with this information and the parameters which are solution of the problem
    INPUT: 
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose between
            ['circle', 'diamond', 'wavy lines', 'iris']
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method - only L-BFGS-B was used
        -name: a name we want for our our files to be save with
        -iris_id: the task to bring from the iris dataset
        -theta: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. Must be an array with shape (qubits, layers, dimension of data)
        -weight: set of parameters needed fot the circuit only if chi == 'weighted_fidelity_chi'. Must be an array with shape (classes, qubits)
        -chi_value: Value of the cost function after minimization
        -acc_train: accuracy for the training set
        -acc_test: accuracy for the test set
        -seed: seed of numpy.random, needed for replicating results
        
    OUTPUT:
        This function has got no outputs, but several files are saved in an appropiate folder. The files are
        -summary.txt: Saves useful information for the problem
        -theta.txt: saves the theta parameters as a flat array
        -alpha.txt: saves the alpha parameters as a flat array
        -weight.txt: saves the weights as a flat array if they exist
    """
    
    foldname = name_folder(chi, problem, qubits, entanglement, layers, method, iris_id)
   
    create_folder(foldname)
    file_text = open(foldname + '/' + name + '_summary.txt','w')
    file_text.write('\nFigur of merit = '+chi)
    file_text.write('\nProblem = ' + problem)
    file_text.write('\nNumber of qubits = ' + str(qubits))
    if qubits != 1:
        file_text.write('\nEntanglement = ' + entanglement)
    file_text.write('\nNumber of layers = ' + str(layers))
    file_text.write('\nMinimization method = '+ method)
    file_text.write('\nRandom seed = '+ str(seed))
    if chi == 'weighted_fidelity_chi':
        file_text.write('\nWEIGHTS = \n')
        file_text.write(str(weights))
    file_text.write('\nchi**2 = ' + str(chi_value))
    file_text.write('\nacc_train = ' + str(acc_train))
    file_text.write('\nacc_test = ' + str(acc_test))
    file_text.close()
    
    np.savetxt(foldname + '/' + name + '_theta.txt', theta.flatten())
    np.savetxt(foldname + '/' + name + '_alpha.txt', alpha.flatten())
    np.savetxt(foldname + '/' + name + '_loss.txt', np.array(loss_history))
    if chi == 'weighted_fidelity_chi':
        np.savetxt(foldname + '/' + name + '_weight.txt', weights.flatten())


def create_folder(directory): 
    """
    Auxiliar function for creating directories with name directory
    
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)


def name_folder(chi, problem, qubits, entanglement, layers, method, iris_id):
    if iris_id is not None:
        chi = chi.lower().replace(' ','_')
        if chi in ['fidelity', 'weighted_fidelity']: chi += '_chi'
        if chi not in ['fidelity_chi', 'weighted_fidelity_chi']:
            raise ValueError('Figure of merit is not valid')
        foldname = chi + '/'
        problem = problem.replace(' ', '_')
        foldname += problem + '/'
        foldname += iris_id + '/'
        foldname += str(qubits) + '_qubits/'
        if qubits != 1: 
            if entanglement.lower()[0] == 'y':
                foldname += 'entangled/'
            if entanglement.lower()[0] == 'n':
                foldname += 'not_entangled/'
            
        foldname += str(layers) + '_layers/'
        foldname += method

    else:
        chi = chi.lower().replace(' ','_')
        if chi in ['fidelity', 'weighted_fidelity']: chi += '_chi'
        if chi not in ['fidelity_chi', 'weighted_fidelity_chi']:
            raise ValueError('Figure of merit is not valid')
        foldname = chi + '/'
        problem = problem.replace(' ', '_')
        foldname += problem + '/'
        foldname += str(qubits) + '_qubits/'
        if qubits != 1: 
            if entanglement.lower()[0] == 'y':
                foldname += 'entangled/'
            if entanglement.lower()[0] == 'n':
                foldname += 'not_entangled/'
            
        foldname += str(layers) + '_layers/'
        foldname += method
    
    return foldname



def painter(chi, problem, qubits, entanglement, layers, method, name, iris_id,
            seed = 30, standard_test = True, samples = 4000, bw = False, err = False):
    """
    This function takes written text files and paint the results of the problem 
    INPUT:
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose among
            ['circle', 'diamond', 'wavy lines', 'iris']
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method - only L-BFGS-B was used
        -iris_id: to treat the iris dataset
        -name: a name we want for our our files to be save with
        -seed: seed of numpy.random, needed for replicating results
        -standard_test: Whether we want to paint the set test used for checking when minimizing. If True, seed and samples are not taken in account
        -samples: number of samples of the test set
        -bw: painting in black and white
    OUTPUT:
        This function has got no outputs, but a file containing the representation of the test set is created
    """
    np.random.seed(seed)
    
    if chi == 'fidelity_chi':
        qubits_lab = qubits
    elif chi == 'weighted_fidelity_chi':
        qubits_lab = 1
        
    if standard_test == True:
        if problem == 'iris':
            (train_data, test_data), drawing = data_generator(problem, iris_id)
    
        else:
            data, drawing = data_generator(problem, iris_id= None)
            test_data = data[20:]
    
        #return test_data
          
    elif standard_test == False:
        test_data, drawing = data_generator(problem, samples = samples)
            
    if problem in ['circle', 'diamond', 'iris', 'iris_ovr']:
        classes = 2

    elif problem == 'wavy lines':
        classes = 4
        
    reprs = representatives(classes, qubits_lab)
    
    params = read_summary(chi, problem, qubits, entanglement, layers, method, name, iris_id)
    
    if chi == 'fidelity_chi':
        theta, alpha = params
        sol_test, acc_test = Accuracy_test(theta, alpha, test_data, reprs, entanglement, chi)
        
    if chi == 'weighted_fidelity_chi':
        theta, alpha, weight = params
        sol_test, acc_test = Accuracy_test(theta, alpha, test_data, reprs,
                                           entanglement, chi, weights = weight)

    foldname = name_folder(chi, problem, qubits, entanglement, layers, method, iris_id)
    samples_paint(problem, drawing, sol_test, foldname, name, bw)


def read_summary(chi, problem, qubits, entanglement, layers, method, name, iris_id):
        
    """
    This function reads the files saved by write_summary and returns theta, alpha and weight parameters
    INPUT: 
        -chi: cost function, to choose between 'fidelity_chi' or 'weighted_fidelity_chi'
        -problem: name of the problem, to choose among
            ['circle', 'diamond', 'wavy lines', 'iris']
        -qubits: number of qubits, must be an integer
        -entanglement: whether there is entanglement or not in the Ansätze, just 'y'/'n'
        -layers: number of layers, must be an integer. If layers == 1, entanglement is not taken in account
        -method: minimization method - only L-BFGS-B was used
        -name: a name we want for our our files to be save with
        
    OUTPUT:
        -theta: set of parameters needed for the circuit. It is an array with shape (qubits, layers, 3)
        -alpha: set of parameters needed for the circuit. It is an array with shape (qubits, layers, dimension of data)
        -weight: set of parameters needed fot the circuit only if chi == 'weighted_fidelity_chi'. It is an array with shape (classes, qubits)
    """
    chi = chi.lower().replace(' ','_')
    if chi in ['fidelity', 'weighted_fidelity']: chi += '_chi'
    if chi not in ['fidelity_chi', 'weighted_fidelity_chi']:
        raise ValueError('Figure of merit is not valid')
    if chi == 'fidelity_chi':
        foldname = name_folder(chi, problem, qubits, entanglement, layers, method, iris_id)
        if problem in ['circle', 'diamond', 'wavy lines', 'iris', 'iris_ovr']:
            theta = np.loadtxt(foldname + '/' + name + '_theta.txt').reshape((qubits, layers, 3))
            dim = 2
            
        alpha = np.loadtxt(foldname + '/' + name + '_alpha.txt').reshape((qubits, layers, dim))
        return theta, alpha
    
    if chi == 'weighted_fidelity_chi':
        foldname = name_folder(chi, problem, qubits, entanglement, layers, method, iris_id)
        if problem in ['circle', 'diamond', 'wavy lines', 'iris', 'iris_ovr']:
            theta = np.loadtxt(foldname + '/' + name + '_theta.txt').reshape((qubits, layers, 3))
            dim = 2
            
        alpha = np.loadtxt(foldname + '/' + name + '_alpha.txt').reshape((qubits, layers, dim))

        if problem == 'wavy lines':
            weight = np.loadtxt(foldname + '/' + name + '_weight.txt').reshape((4, qubits))
        if problem in ['circle','diamond', 'iris', 'iris_ovr']:
            weight = np.loadtxt(foldname + '/' + name + '_weight.txt').reshape((2, qubits))
        return theta, alpha, weight

def samples_paint(problem, settings, sol, foldname, filename, bw):
    """
    This function takes the problem and the points when they are already classified, and saves a picture of them
    INPUT: 
        -problem: name of the problem, to choose among
            ['circle', 'diamond', 'wavy lines', 'iris']
        -settings: parameters the function needs for drawing. Provided by problem_gen.problem_gen
        -sol: solutions of the points alreafy classified
        -foldname : name of the folder where we store results
        -filename: name of the files we will produce
        -bw: black and white, True/False
    OUTPUT:
        a file with the points and their classes, and whether they are right or wrong
    """
    if bw == False:
        colors_classes = get_cmap('plasma')
        norm_class = Normalize(vmin=-.5,vmax=np.max(sol[:,-3]) + .5)
    
        colors_rightwrong = get_cmap('RdYlGn')
        norm_rightwrong = Normalize(vmin=-.1,vmax=1.1)
        
    if bw == True:
        colors_classes = get_cmap('Greys')
        norm_class = Normalize(vmin=-.1,vmax=np.max(sol[:,-3]) + .1)
    
        colors_rightwrong = get_cmap('Greys')
        norm_rightwrong = Normalize(vmin=-.1,vmax=1.1)

    fig, axs = plt.subplots(ncols = 2, figsize=(10,5))
    ax = axs[0]
    if problem in ['circle']:
        centers, radii = settings
        for c, r in zip(centers, radii):
            ca = plt.Circle(c, r, color='k', fill=False, linewidth=2)
            ax.add_artist(ca)
    
    elif problem == 'wavy lines':
        freq = settings
        s = np.linspace(-1,1,100)
        ax.plot(s, np.clip(s + np.sin(freq * np.pi * s), -1, 1), 'k-')
        ax.plot(s, -s + np.sin(freq * np.pi * s), 'k-')

    elif problem == 'diamond':
        limit, scale = settings
        
        s = np.linspace(-1,1,200)
        for k in range(-10,10):        
            ax.plot(s, np.clip(k*scale-s, -1, 1), 'k-', linewidth = 1)
            ax.plot(s, np.clip(s-k*scale+scale/2, -1, 1), 'k-', linewidth = 1)
    
    elif problem in ['iris', 'iris_ovr']:
        pass
              
    if problem in ['iris', 'iris_ovr']:
        ax.scatter(sol[:,0], sol[:,1], c=sol[:,-2], cmap = colors_classes, s=30, norm=norm_class)
    else:
        ax.scatter(sol[:,0], sol[:,1], c=sol[:,-2], cmap = colors_classes, s=2, norm=norm_class)
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.tick_params(axis='both',labelsize=16)
    if problem in ['iris','iris_ovr']:
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
    else:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    ax.margins(0)
    ax.set_aspect('equal', adjustable='box')

    bx = axs[1]    
    if problem in ['iris', 'iris_ovr']:
        bx.scatter(sol[:,0], sol[:,1], c=sol[:,-1], cmap = colors_rightwrong, s=30, norm=norm_rightwrong)
    else:
        bx.scatter(sol[:,0], sol[:,1], c=sol[:,-1], cmap = colors_rightwrong, s=2, norm=norm_rightwrong)
  
    if problem == 'circle':
        centers, radii = settings
        for c, r in zip(centers, radii):
            ca = plt.Circle(c, r, color='k', fill=False, linewidth=2)
            bx.add_artist(ca)
    
    elif problem == 'wavy lines':
        freq = settings
        s = np.linspace(-1,1,100)
        bx.plot(s, np.clip(s + np.sin(freq * np.pi * s), -1, 1), 'k-')
        bx.plot(s, -s + np.sin(freq * np.pi * s), 'k-')

    elif problem == 'diamond':
        limit, scale = settings
        s = np.linspace(-1,1,100)
        for k in range(-6,6):        
            bx.plot(s, np.clip(k*scale-s, -1, 1), 'k-', linewidth = 1)
            bx.plot(s, np.clip(s-k*scale+scale/2, -1, 1), 'k-', linewidth = 1)
        
    elif problem in ['iris','iris_ovr']:
        pass

    
    bx.set_xlabel('x', fontsize=16)
    bx.tick_params(axis='x', labelsize = 16)
    bx.tick_params(axis='y', labelsize=0)
    if problem in ['iris','iris_ovr']:
        bx.set_xlim(-4, 4)
        bx.set_ylim(-4, 4)
    else:
        bx.set_xlim(-1, 1)
        bx.set_ylim(-1, 1)
    bx.margins(0)
    bx.set_aspect('equal', adjustable='box')
