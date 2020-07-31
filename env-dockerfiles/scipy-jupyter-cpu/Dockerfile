FROM jupyter/scipy-notebook
LABEL maintainer "Corey Ryan Hanson"

#Installing libraries not included in the Jupyter scipy notebook ://hub.docker.com/r/jupyter/scipy-notebook/

RUN conda update --all --quiet --yes && \
    conda clean --all -f -y && \
    npm cache clean --force && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

RUN conda install --quiet --yes -c conda-forge \
    'imbalanced-learn' \
    'nltk' \
    'PyGithub' \
    'scikit-surprise ' \
    'selenium' \
    'scrapy' \
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

# This block contains dependencies required to fix broken pip install of fbprophet.
RUN pip install --no-cache-dir \
    'convertdate' \
    'LunarCalendar' \
    'holidays' \
    'pystan'

# Packages that won't cleanly install through Anaconda.
RUN pip install --no-cache-dir \
    'fbprophet' \
    'spotipy' \
    'stravalib' \
    'xgboost'

WORKDIR $HOME/work

CMD ["start-notebook.sh"]
