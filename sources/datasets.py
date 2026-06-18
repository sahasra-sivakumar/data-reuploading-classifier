import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#=======================================================
#===============DATA GENERATION FUNCTION================
#=======================================================
problems = ['circle', 'diamond', 'wavy lines', 'iris']

def data_generator(problem, iris_id, samples=None):
    """
    This function generates the data for a problem
    INPUT: 
        -problem: Name of the problem, one of: 'circle', 'diamond', 'wavy lines', 'iris'
        -samples Number of samples for the data
    OUTPUT:
        -data: set of training and test data
        -settings: things needed for drawing
    """
    problem = problem.lower()
    if problem not in problems:
        raise ValueError('problem must be one of {}'.format(problems))
    if samples == None:
        samples = 4200
            
    if problem == 'circle':
        data, settings = circle(samples)
        return data, settings 
    
    elif problem == 'diamond':
        data, settings = diamond(samples)
        return data, settings 
    
    elif problem == 'wavy lines':
        data, settings = wavy_lines(samples)
        return data, settings 
    
    elif problem == 'iris':
        data_train, data_test, settings = get_iris_binary_print(iris_id) 
        return (data_train, data_test), settings
    
#=======================================================
#=================SYNTHETIC DATASETS====================
#=======================================================

#Cirle's generation function (obtained from the article)
def circle(samples):
    #np.random.seed(random_seed)
    centers = np.array([[0, 0]])
    radii = np.array([np.sqrt(2/np.pi)]) #def radius so it represents 0.5 the area of the square - should be a balanced dataset in classification
    data=[]
    dim = 2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1 #so that every point its inside the squared limit (-1,1)
        #random.rand = create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1)
        y = 0
        for c, r in zip(centers, radii):  
            if np.linalg.norm(x - c) < r: #calculates distance between point and center to classify if inside the circle or not
                y = 1 
        
        data.append([x, y])
            
    return data, (centers, radii) 

#Circle's visualization function
def plot_circle(data, centers, radii):
    
    x = np.array([i[0] for i in data]) #datapoints
    y = np.array([i[1] for i in data]) #labels definition
    
    center = centers[0]
    radius = radii[0]
    
    plt.figure(figsize=(6,6))
    plt.scatter(x[y == 0, 0], x[y == 0, 1], c='goldenrod', alpha=0.5, s=20, label='Outside') #plotting the dataset with its labels
    plt.scatter(x[y == 1, 0], x[y == 1, 1], c='fuchsia', alpha=0.5, s=20, label='Inside')
    
    theta = np.linspace(0, 2*np.pi, 200)
    plt.plot(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta),  #plotting the circle
            'k--', linewidth=2, label='Border')    
    
data, (centers, radii) = circle(samples=4200) #running for N=4200
fig = plot_circle(data, centers, radii)
#plt.show()

#=======================================================

#Diamond's shape pattern generation function
def diamond(samples, random_seed=30):
    np.random.seed(random_seed)
    data = []
    limit = 1
    scale = 0.5

    x_i = np.random.uniform(-1, limit, samples) #generates the data points inside the square limit
    y_i = np.random.uniform(-1, limit, samples)
    x = np.column_stack([x_i, y_i])

    for i in range(len(x)):
        logic = (np.floor((x[i][0] + x[i][1]) / scale).astype(int) + np.floor((x[i][0] - x[i][1] + scale/2) / scale).astype(int)) % 2  
        data.append((x[i], int(logic)))
    
    #(x + y) and (x - y) to create diagonals
    #Default function: (floor((x+y)/s) + floor((x-y+s/2)/s)) mod 2
    #scale determines the frequence of pattern - if bigger, diamonds get bigger; otherwise, denser
    return data, (limit, scale)

#Diamond's visualization function
def plot_diamond(data):
    x = np.array([data[i][0] for i in range(len(data))])
    y = np.array([data[i][1] for i in range(len(data))])
    colors = np.where(y == 0, 'fuchsia', 'goldenrod')

    plt.figure(figsize=(6, 6))
    plt.scatter(x[:,0], x[:,1], c=colors, s=10)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect('equal')
    plt.title("Diamonds Pattern Classification")
    
data, _ = diamond(samples=4200)
fig = plot_diamond(data)
#plt.show()

#=======================================================
#Wavy pattern generation function (obtained from the article)
def wavy_lines(samples, freq = 1):
    def fun1(s):
        return s + np.sin(freq * np.pi * s)
    
    def fun2(s):
        return -s + np.sin(freq * np.pi * s)
    data=[]
    dim=2
    for i in range(samples):
        x = 2 * (np.random.rand(dim)) - 1
        if x[1] < fun1(x[0]) and x[1] < fun2(x[0]): y = 0
        if x[1] < fun1(x[0]) and x[1] > fun2(x[0]): y = 1
        if x[1] > fun1(x[0]) and x[1] < fun2(x[0]): y = 2
        if x[1] > fun1(x[0]) and x[1] > fun2(x[0]): y = 3        
        data.append([x, y])

    return data, freq

#Wavy pattern visualization function
def plot_wavy(data,freq):
#separating into features and labels
    x = np.array([i[0] for i in data]) 
    y = np.array([i[1] for i in data])

    plt.figure(figsize=(6,6))

    plt.scatter(x[:,0], x[:,1], c=y, cmap='spring', s=10)

# curves
    s = np.linspace(-1, 1, 500)
    plt.plot( s, np.clip(s + np.sin(freq * np.pi * s), -1, 1),   'k-', linewidth=2)
    plt.plot( s, -s + np.sin(freq * np.pi * s), 'k-', linewidth=2)
#  visual adjustments
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.gca().set_aspect('equal')
    plt.grid(alpha=0.3)
    plt.title('Wavy Lines Classification')

data, freq = wavy_lines(samples=4200, freq=1)
fig = plot_wavy(data,freq)
#plt.show()

#=======================================================
#======================IRIS DATASET=====================
#=======================================================
# 100 datapoints
def get_iris_binary_print(iris_id, test_size=0.5, random_state=42):    
    #Load Data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    #Binary Classification (0:setosa, 1:versicolor, 2:virginica)
    if iris_id == "setosa_vs_versicolor":   
        mask = (y == 0) | (y == 1)
        X, y = X[mask], y[mask]

    elif iris_id == "versicolor_vs_virginica":
        mask = (y == 1) | (y == 2)
        X, y = X[mask], y[mask]
        y = np.where(y == 1, 0, 1) 

    elif iris_id == "setosa_ovr":
        y = np.where(y == 0, 1, 0)  # setosa->1, rest->0

    elif iris_id == "versicolor_ovr":
        y = np.where(y == 1, 1, 0)  # versicolor->1, rest->0

    elif iris_id == "virginica_ovr":
        y = np.where(y == 2, 1, 0)  # virginica->1, rest->0

    else:
        raise ValueError("iris_id must be one of: setosa_vs_versicolor, "
                         "versicolor_vs_virginica, setosa_ovr, "
                         "versicolor_ovr, virginica_ovr")

    
    #Reduce to 2 features (PCA) so we can plot decision boundaries later
    from sklearn.decomposition import PCA

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify= y)

    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)   # fit only on train
    X_test  = pca.transform(X_test)        # apply same transform to test

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    data_train = [[(x[0], x[1]), l] for x, l in zip(X_train, y_train)]
    data_test = [[(x[0], x[1]), l] for x, l in zip(X_test, y_test)]
    
    return data_train, data_test, len(X)

def plot_dataset(data_train, data_set):
    """
    Plot the 2D PCA representation of the dataset.
    """
    data = data_train + data_set
    X = np.array([point[0] for point in data])  # shape (n, 2)
    y = np.array([point[1] for point in data])  # shape (n,)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', s=80)
    plt.title("Iris Dataset - 2D Representation")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label="Class label")
    plt.show()
