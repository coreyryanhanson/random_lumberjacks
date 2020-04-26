# Python environment dockerfiles

A base set of docker build commands that can be applied to several different base images at once with the included pyfile to create the necessary dockerfile.

CUDA and CUDA-less versions are included.

## A few handy volume flags
#### Note: multiple options are listed. Choose according to base image

### Mount a shared directory to save work to
```
-v "$PWD":/home/anacuda/work
-v "$PWD":/home/jovyan/work
```
This will launch jupyter from your present directory. Change from ```$PWD``` for another location.
### Save your jupyter notebook config settings
```
-v /home/<your_username>/.jupyter:/home/anacuda/.jupyter
-v /home/<your_username>/.jupyter:/home/jovyan/.jupyter
```
### Save external data loaded from NLTK
```
-v /home/<your_username>/nltk_data:/home/anacuda/nltk_data
-v /home/<your_username>/nltk_data:/home/jovyan/nltk_data
```
### Note: I have this set up so that in order to use selenium, you would run the drivers seperately multiple container setup using docker-compose or kubernetes.
I will post a setup yaml in the future, once I learn how to get that running on RHEL's podman.
