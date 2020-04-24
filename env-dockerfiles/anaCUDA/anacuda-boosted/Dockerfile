FROM coreyhanson/anacuda:runtime
LABEL maintainer "Corey Ryan Hanson"

#Installing libraries not included in the Jupyter scipy notebook ://hub.docker.com/r/jupyter/scipy-notebook/

RUN conda install --quiet --yes \
    'imbalanced-learn=0.6.2' \
    'selenium=3.141.0' && \
    conda update --all --quiet --yes && \
    conda clean --all -f -y && \
    npm cache clean --force && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

RUN pip install \
    'spotipy' \
    'xgboost'

WORKDIR $HOME/work

CMD ["start-notebook.sh"]
