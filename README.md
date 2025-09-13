
<h3 align="center">G3LHalo</h3>

<p align="center">
    Code for calculating galaxy-matter power- and bispectra using the halo model
</p>

<!-- ABOUT THE PROJECT -->
## About The Project

This code uses the halo model and HODs to model the power- and bispectra of galaxy- and matter distributions. For an application see <a href="https://ui.adsabs.harvard.edu/abs/2022A%26A...665A..38L/abstract"> Linke, Simon, Schneider + (2023) </a>.


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
* **python3**: This code will not work for python2! 
* **pyccl**: Check <a href="https://ccl.readthedocs.io/en/latest/"> here </a> for how to install it
* **numpy**
* **scipy**
* For example notebooks: **matplotlib**

### Installation

To install this code, clone the gitrepo, go to the root folder and execute
```
pip install .
```

<!-- USAGE EXAMPLES -->
## Usage

The folder `exampleNotebooks` contains examples for how to define the halo model ingredients, compute 3D power and bispectra, and limber-integrated power and bispectra.
The examples use functions defined in the `pyccl`, but in principle a user can provide any function for the halo mass function, halo bias and halo occupation distributions. 

<!-- LICENSE -->
## License

Distributed under the GNU General Public License v 3.0.

Please cite <a href="https://ui.adsabs.harvard.edu/abs/2022A%26A...665A..38L/abstract">  Linke et al.(2023) </a> if you use the code for a publication. 

<!-- CONTACT -->
## Contact

Laila Linke - [laila.linke@uibk.ac.at](mailto:laila.linke@uibk.ac.at)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* The code is based on code by Jens Rödiger and his <a href="https://hdl.handle.net/20.500.11811/4086"> PhD Thesis </a> .
* This ReadMe is based on <a href="https://github.com/othneildrew/Best-README-Template"> Best-README-Template </a>

