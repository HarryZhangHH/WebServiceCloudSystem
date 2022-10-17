# wscs4_assignment4 - Visualization package

![compute package workflow](https://github.com/straightedge77/wscs4_visualize/actions/workflows/main.yml/badge.svg) [![DOI](https://zenodo.org/badge/498728336.svg)](https://zenodo.org/badge/latestdoi/498728336)

## Code Introduction
- visualize.py is the python code file to run, it contains different visualization functions of the titanic dataset analysis
- container.yml is the file describing how to interface with the code and how to build package
- test.txt contains the branescript used for unit tests of the package
- brane is the compiled cli binary file, used for automated build
- data folder is the folder which brane filesystem mounted on when run package
- test foler is the folder which brane filesystem mounted  on when test package
- .github/workflows/main.yml contains the command used by github in the automated build and test of the package
- workflow.ipynb contains the branescript pipeline

## Prerequisites
The package can run both in local and Kubernetes infrastructures. To test it, brane cli need to be installed. To run the workflow.ipynb the instance must be  deployed and brane-ide must be installed.

## Run
The brane can be built using command(in the root repository)
```
brane build ./container.yml
```
It can also be imported using command
```
brane import straightedge77/wscs4_visualize
```
You test each function of the brane package using command
```
brane test --data ./data visualize
```
You can also perform unit test using command
```
brane import straightedge77/wscs4_test
brane run test.txt --data ./test
```
You run the package in the brane instance using command
```
brane login http://<IP> --username <user>
brane push visualize
brane repl --remote http://<IP>
```
Replace <IP> with 127.0.0.1 if you use local infrastructure. Replace <IP> with the remote brane instance IP address when you use Kubernetes infrastructures. Replace <user> with a name of your choosing.  
You can also run the pipeline using jupyterlab designed for brane. To use that follow the instruction on https://github.com/epi-project/brane-ide. Note that, the packages used still need to be pushed first.
When you try to run the pipeline using jupyterlab on Kubernets. Change the 127.0.0.1 in the docker-compose.yml to 0.0.0.0 (under the ports key for the container). And run this command:
```
make start-ide BRANE_DRV="<IP>:50053" BRANE_MOUNT_DFS="redis://<IP>"
```
Replace <IP> with the remote brane instance address. Finally, don't forget to replace the 127.0.0.1 IP in the resulting URL with the one of your cluster before copying it in the browser.

## Reference
https://wiki.enablingpersonalizedinterventions.nl/user-guide/welcome.html  
https://wiki.enablingpersonalizedinterventions.nl/admins/welcome.html  
https://zenodo.org/  
https://github.com/epi-project/brane-ide  
https://github.com/features/actions  
https://www.kaggle.com/c/titanic/overview  
https://guides.lib.berkeley.edu/citeyourcode  
https://kubernetes.io/docs/tasks/configure-pod-container/configure-persistent-volume-storage/  
https://computingforgeeks.com/restrict-kubernetes-service-account-users-to-a-namespace-with-rbac/  
https://www.base64encode.org/  
https://www.kaggle.com/code/zocainviken/titanic-disaster-deeper-into-dataset