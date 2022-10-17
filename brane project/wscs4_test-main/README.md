# wscs4_assignment4 - Test package

[![DOI](https://zenodo.org/badge/499092194.svg)](https://zenodo.org/badge/latestdoi/499092194)

## Code Introduction
- compare.py is the python code file to run, it contains different functions of testing 
- container.yml is the file describing how to interface with the code and how to build package

## Prerequisites
The package can run both in local and Kubernetes infrastructures. To run it, brane cli need to be installed. To run it on the brane instance, instance must be deployed.

## Run
The package is mainly used to perform unit tests on the brane compute and visualization package. Therefore, you only need to build it or import it.  
The brane can be built using command(in the root repository)
```
brane build ./container.yml
```
It can also be imported using command
```
brane import straightedge77/wscs4_test
```
