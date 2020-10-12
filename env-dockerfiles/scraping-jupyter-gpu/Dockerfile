FROM coreyhanson/anacuda-scipy
LABEL maintainer "Corey Ryan Hanson"

#Installing libraries not included in the Jupyter scipy notebook ://hub.docker.com/r/jupyter/scipy-notebook/

RUN source activate $CONDA_ROOT && \
    conda install --quiet --yes -c conda-forge \
    'nest-asyncio' \
    'PyGithub' \
    'pyproj' \
    'selenium' \
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
    'pycoingecko' \
    'spotipy' \
    'stravalib'

RUN source activate $CONDA_ROOT && \
    pip install --no-cache-dir --user --upgrade \
    git+https://github.com/twintproject/twint.git@origin/master#egg=twint

WORKDIR $HOME/work

CMD ["start-notebook.sh"]

# Customizations for Jupyter notebook.
RUN jupyter labextension install @jupyterlab/toc --no-build && \
    jupyter labextension install @aquirdturtle/collapsible_headings --no-build && \
    jupyter labextension install @lckr/jupyterlab_variableinspector --no-build && \
    jupyter labextension install jupyterlab-theme-solarized-dark --no-build && \
    jupyter labextension install @ijmbarr/jupyterlab_spellchecker --no-build && \
    jupyter lab build -y && \
    jupyter lab clean -y && \
    npm cache clean --force && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
