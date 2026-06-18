# Data Re-uploading for a Universal Quantum Classifier — Qiskit Reproduction

A reproduction study of the paper **"Data Re-uploading for a Universal Quantum Classifier"** by Pérez-Salinas et al. (2020), implemented in Python and Qiskit for the Laboratory of Computational Physics Project.

> Pérez-Salinas, A., Cervera-Lierta, A., Gil-Fuster, E., & Latorre, J. I. (2020).  
> *Data re-uploading for a universal quantum classifier.*  
> **Quantum, 4**, 226. https://doi.org/10.22331/q-2020-02-06-226

---

## 👥 Group Members

- **Elena Niero**
- **Eylul Cagla**
- **Melissa Nespeque**
- **Sahasra Sivakumar**

---

## 🎯 Project Objective

This project reproduces and analyzes the quantum classification framework proposed in the original paper, which introduces the concept of **data re-uploading**: a technique that embeds classical input data multiple times throughout a variational quantum circuit, enabling a single qubit to act as a universal classifier and compare its performance with respect to classical classifiers.

The key idea is that a qubit's state is manipulated by successive single-qubit unitary operations of the form:

$$U(\vec{x}, \vec{\theta}) = U_L(\vec{x}, \vec{\theta}_L) \cdots U_2(\vec{x}, \vec{\theta}_2)\, U_1(\vec{x}, \vec{\theta}_1)$$

where each layer re-encodes the input data $\vec{x}$ alongside trainable parameters $\vec{\theta}_l$, allowing the circuit to learn arbitrarily complex decision boundaries.

The goals of this reproduction are to:
- Implement single-qubit and multi-qubit classifiers using Qiskit
- Implement classical classifier algorithms (such as Linear Regression, XGBoost...)
- Train them on synthetic 2D binary classification datasets and on a real life dataset (Iris)
- Evaluate accuracy as a function of the number of layers and qubits
- Compare results within the circuits and datasets
 
---


## 📄 Reference

```bibtex
@article{Perez-Salinas2020,
  title   = {Data re-uploading for a universal quantum classifier},
  author  = {Pérez-Salinas, Adrián and Cervera-Lierta, Alba and Gil-Fuster, Elies and Latorre, José I.},
  journal = {Quantum},
  volume  = {4},
  pages   = {226},
  year    = {2020},
  doi     = {10.22331/q-2020-02-06-226}
}
```
