FROM jupyter/pyspark-notebook
LABEL maintainer "Corey Ryan Hanson"

#Installing libraries not included in the Jupyter scipy notebook ://hub.docker.com/r/jupyter/scipy-notebook/

# Base files accessible all environments

RUN source activate $CONDA_ROOT && \
    conda install --quiet --yes -c conda-forge \
    'imbalanced-learn' \
    'fbprophet' \
    'fiona' \
    'gensim' \
    'geopandas' \
    'geoplot=0.4.1' \
    'hyperopt' \
    'libnetcdf' \
    'nltk' \
    'plotly' \
    'pyproj' \
    'scikit-surprise ' \
    'shapely' \
    'spacy' \
    'textblob ' \
    'wordcloud' && \
    conda clean --all -f -y && \
    npm cache clean --force && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Packages that won't cleanly install through Anaconda.
RUN source activate $CONDA_ROOT && \
    pip install --no-cache-dir \
    'xgboost'

# Additional pip packages for spaCy models
RUN source activate $CONDA_ROOT && \
    python -m spacy download en_core_web_sm

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
