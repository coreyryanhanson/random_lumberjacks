FROM coreyhanson/anacuda-scipy
LABEL maintainer "Corey Ryan Hanson"

#Installing libraries not included in the Jupyter scipy notebook ://hub.docker.com/r/jupyter/scipy-notebook/

RUN source activate $CONDA_ROOT && \
    conda install --quiet --yes -c conda-forge \
    'nest-asyncio' \
    'psycopg2' \
    'PyGithub' \
    'pyproj' \
    'selenium' \
    'shapely' \
    'scrapy' \
    'tweepy' && \
    conda clean --all -f -y && \
    npm cache clean --force && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Packages that won't cleanly install through Anaconda.
RUN source activate $CONDA_ROOT && \
    pip install --no-cache-dir \
    'geoalchemy2' \
    'ipython-sql' \
    'praw' \
    'pycoingecko' \
    'spotipy' \
    'stravalib'

RUN source activate $CONDA_ROOT && \
    pip install --no-cache-dir --user --upgrade \
    git+https://github.com/twintproject/twint.git@origin/master#egg=twint

WORKDIR $HOME/work

CMD ["start-notebook.sh"]

# Customizations for Jupyter notebook.

RUN source activate base && \
    conda install --quiet --yes \
    'ipywidgets' \
    'widgetsnbextension' && \
    conda clean --all -f -y && \
    jupyter nbextension enable --py widgetsnbextension --sys-prefix && \
    # Also activate ipywidgets extension for JupyterLab
    # Check this URL for most recent compatibilities
    # https://github.com/jupyter-widgets/ipywidgets/tree/master/packages/jupyterlab-manager
    jupyter labextension install @jupyter-widgets/jupyterlab-manager \
                                 jupyter-matplotlib \
                                 @aquirdturtle/collapsible_headings \
                                 jupyterlab-theme-solarized-dark \
                                 @ijmbarr/jupyterlab_spellchecker && \
    jupyter lab clean -y && \
    npm cache clean --force && \
    rm -rf "/home/${NB_USER}/.cache/yarn" && \
    rm -rf "/home/${NB_USER}/.node-gyp" && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"
