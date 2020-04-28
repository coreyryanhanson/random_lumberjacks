FROM jupyter/tensorflow-notebook
LABEL maintainer "Corey Ryan Hanson"

#Installing libraries not included in the Jupyter scipy notebook ://hub.docker.com/r/jupyter/scipy-notebook/

RUN conda install --quiet --yes \
    'imbalanced-learn' \
    'nltk' \
    'plotly' \
    'selenium' \
    'scikit-surprise ' \
    'scrapy' \
    'wordcloud' && \
    conda update --all --quiet --yes && \
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

RUN pip install --no-cache-dir \
    'spotipy' \
    'xgboost'

WORKDIR $HOME/work

CMD ["start-notebook.sh"]

RUN conda install --quiet --yes \
    'pillow' && \
    conda update --all --quiet --yes && \
    conda clean --all -f -y && \
    jupyter lab build -y && \
    jupyter lab clean -y && \
    npm cache clean --force && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER
