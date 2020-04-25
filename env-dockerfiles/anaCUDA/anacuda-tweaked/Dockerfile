FROM coreyhanson/anacuda:boosted
LABEL maintainer "Corey Ryan Hanson"

#Convienience/theme tweaks for the image.

RUN conda update --all --quiet --yes && \
    conda clean --all -f -y && \
    jupyter labextension install @jupyterlab/toc --no-build && \
    jupyter labextension install @aquirdturtle/collapsible_headings --no-build && \
    jupyter labextension install @lckr/jupyterlab_variableinspector --no-build && \
    jupyter labextension install @ijmbarr/jupyterlab_spellchecker --no-build && \
    jupyter lab build -y && \
    jupyter lab clean -y && \
    npm cache clean --force && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER && \
    rm -rf /home/$NB_USER/*

