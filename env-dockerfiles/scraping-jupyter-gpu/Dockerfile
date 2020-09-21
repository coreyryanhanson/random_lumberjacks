FROM coreyhanson/anacuda-scipy
LABEL maintainer "Corey Ryan Hanson"

#Installing libraries not included in the Jupyter scipy notebook ://hub.docker.com/r/jupyter/scipy-notebook/

RUN source activate $CONDA_ROOT && \
    conda install --quiet --yes -c conda-forge \
    'pycoingecko' \
    'PyGithub' \
    'pyproj' \
    'selenium' \
    'scrapy' && \
    conda clean --all -f -y && \
    npm cache clean --force && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Packages that won't cleanly install through Anaconda.
RUN source activate $CONDA_ROOT && \
    pip install --no-cache-dir \
    'spotipy' \
    'stravalib'

WORKDIR $HOME/work

CMD ["start-notebook.sh"]
