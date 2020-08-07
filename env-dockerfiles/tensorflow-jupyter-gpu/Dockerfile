FROM coreyhanson/anacuda-tensorflow
LABEL maintainer "Corey Ryan Hanson"

#Installing libraries not included in the Jupyter scipy notebook ://hub.docker.com/r/jupyter/scipy-notebook/

RUN conda install --quiet --yes -c conda-forge \
    'fbprophet' \
    'fiona' \
    'geopandas' \
    'geoplot' \
    'nltk' \
    'pyproj' \
    'scikit-surprise ' \
    'shapely' \
    'wordcloud' && \
    conda install --quiet --yes -c plotly \
    'plotly' && \
    conda clean --all -f -y && \
    jupyter labextension install @jupyterlab/toc --no-build && \
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

# Packages that won't cleanly install through Anaconda.
RUN pip install --no-cache-dir \
    'xgboost'

WORKDIR $HOME/work

CMD ["start-notebook.sh"]

RUN conda install --quiet --yes \
    'pillow' && \
    conda install --quiet --yes -c orbingol geomdl && \
    conda clean --all -f -y && \
    jupyter lab build -y && \
    jupyter lab clean -y && \
    npm cache clean --force && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
