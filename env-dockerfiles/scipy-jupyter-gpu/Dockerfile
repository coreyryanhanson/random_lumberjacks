FROM coreyhanson/anacuda-scipy
LABEL maintainer "Corey Ryan Hanson"

#Installing libraries not included in the Jupyter scipy notebook ://hub.docker.com/r/jupyter/scipy-notebook/

# Base files accessible all environments

RUN source activate $CONDA_ROOT && \
    conda install --quiet --yes -c conda-forge \
    'imbalanced-learn' \
    'fbprophet' \
    'fiona' \
    'geopandas' \
    'geoplot=0.4.1' \
    'hyperopt' \
    'libnetcdf' \
    'plotly' \
    'psycopg2' \
    'pyproj' \
    'scikit-surprise ' \
    'shapely'  && \
    conda clean --all -f -y && \
    npm cache clean --force && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Packages that won't cleanly install through Anaconda.
RUN source activate $CONDA_ROOT && \
    pip install --no-cache-dir \
    'ipython-sql' \
    'xgboost'

# NLP modules
RUN source activate $CONDA_ROOT && \
    conda install --quiet --yes -c conda-forge \
    'gensim' \
    'nltk' \
    'pyldavis' \
    'spacy' \
    'textblob ' \
    'wordcloud' && \
    conda clean --all -f -y && \
    npm cache clean --force && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

RUN source activate $CONDA_ROOT && \
    pip install --no-cache-dir \
    'better_profanity==0.6.1' \
    'pycontractions' \
    'spacymoji'

# Additional pip packages for spaCy models
RUN source activate $CONDA_ROOT && \
    python -m spacy download en_core_web_sm

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
