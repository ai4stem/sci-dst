# **SCI-DST: Critical Dynamics in Creative Cognition**

This repository contains the Python implementation of the Socio-Cognitive Ising–Dynamic Systems (SCI-DST) framework for modelling creative cognition, as described in our paper:

Jho, H. & Luo, W. (Year). Critical Dynamics in Creative Cognition: An Ising-Based Computational Framework for Modelling Insight. \[Journal Name \- e.g., Scientific Reports\], \[Volume\], \[Article Number\]. \[DOI Link \- when available\]\*  
(Please update the citation details upon publication)

## **Overview**

The SCI-DST framework models creative insight by integrating two-state Ising dynamics with sociocultural fields and dynamic systems principles. This repository provides implementations for:

1. **Traditional 2D Lattice-Based Ising Model**: A baseline model using a regular grid structure.  
2. **Small-World Network Ising Model**: An extended model using the Watts-Strogatz network topology, better reflecting neural connectivity patterns.

The code simulates key phenomena including phase transitions, divergent-convergent cycles (creativity cycles), the effect of associative structures on dynamics (modelling individual differences), and comparison with constant-temperature models (Boltzmann Machines).

## **Requirements**

The code requires Python 3 and the following packages:

* NumPy  
* NetworkX  
* Matplotlib  
* tqdm (for progress bars)  
* pandas (for heatmap analysis)

You can install dependencies using pip:

pip install numpy networkx matplotlib tqdm pandas

*(Optionally, provide a requirements.txt file)*

## **Code Structure Overview**

The repository includes:

* Core model implementations (lattice\_ising.py, smallworld\_ising.py).  
* Scripts to run different simulation analyses (run\_\*.py).  
* Plotting utilities (e.g., plotting.py, methods within model classes).

## **Running Simulations and Reproducing Figures**

You can run the different analyses presented in the paper using the provided run\_\*.py scripts. The main analyses correspond to the manuscript figures as follows:

* **Figure 1 (Lattice Phase Transition):** Run the 2D lattice simulation script.  
  python run\_lattice\_simulation.py

  (Outputs lattice\_results.png)  
* **Figure 2 & 3 (Small-World Scan & Cycle):** Run the basic small-world simulation script.  
  python run\_smallworld.py

  (Outputs smallworld\_scan\_results.png, smallworld\_cycle\_results.png, etc.)  
  Note: Ensure parameters like t\_low in this script match the final manuscript values.  
* **Figure 4 (Individual Differences):** Run the individuality comparison script.  
  python run\_individuality\_sim.py

  (Outputs individuality\_comparison\_cycles.png)  
* **Figure 6 (Comparison vs. BM):** Run the Boltzmann Machine comparison script.  
  python run\_comparison\_sim.py

  (Outputs comparison\_sci\_dst\_vs\_bm.png)  
* **Parameter Sweep Heatmap:** Run the heatmap generation script.  
  python run\_heatmap.py

  (Generates data and contains code to plot heatmap\_rho\_0.0005.png).

Results (plots) will typically be saved as PNG files in the same directory.

## **Parameter Sweep Analysis (Heatmap)**

*(Content adapted from former Appendix B)*

To understand how key parameters jointly influence creative yield, we performed a parameter sweep.

* **Objective:** Quantify how joint creative yield—defined conceptually as the product of novelty (approximated by Shannon entropy S) and usefulness (approximated by the inverse of convergence time)—varies across a grid of temperature (T), learning rate (η), and decay constant (ρ).  
* **Grid:**  
  * T∈1.5,2.0,2.25,2.5,3.0,5.0  
  * η∈0.001,0.005,0.01  
  * ρ∈0.0001,0.0005,0.001  
* **Simulation & Visualisation:** The run\_heatmap.py script contains the logic to explore this parameter space. An example heatmap for ρ=0.0005 shows creative yield across the T×η plane:

  (Assumption: This heatmap image is generated and available in the repository)  
* **Findings Summary:**  
  1. Yield peaks generally form near the critical temperature (T≈2.25−2.75 in the small-world model), with optimal η depending on T.  
  2. Higher decay ρ can broaden the optimal parameter range.  
  3. The region of high creative yield aligns conceptually with the system operating near criticality, balancing exploration and exploitation.

## **Implementation Details Summary**

* **Network Structure:** The lattice model uses a fixed grid; the small-world model uses a Watts-Strogatz graph for more realistic connectivity.  
* **Distance:** Calculated via grid units (lattice) or network path length (small-world).  
* **Susceptibility:** Estimated in the small-world model via field perturbations.

## **Citation**

If you use this code or the SCI-DST framework in your research, please cite our paper:

@article{JhoLuo2025SciRep,  
  title     \= {Critical Dynamics in Creative Cognition: An Ising-Based Computational Framework for Modelling Insight},  
  author    \= {Jho, Hunkoog and Luo, Wei},  
  journal   \= {Scientific Reports},  
  year      \= {2025},  
  volume    \= {},  
  number    \= {},  
  pages     \= {},  
  doi       \= {},  
  publisher \= {Nature Publishing Group}  
}  
\# Please update volume, number, pages, doi upon publication.

## **License**

*(Optional: Add license information, e.g., MIT License)*