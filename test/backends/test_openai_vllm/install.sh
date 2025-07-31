#!/bin/bash -xe

conda env remove -y -n mellea || true
conda env create -f $(readlink -ef $(dirname $0))/environment.yml

in-conda (){
    conda run -n mellea $@
}


in-conda uv pip install -e .[dev]
in-conda uv pip install pre-commit
# in-conda pre-commit install


install-vllm-fork (){

    # first, install vllm
    uv pip install vllm==0.9.1

    # copying the shared objects that are missing in the custom build
    rsync -av --prune-empty-dirs --include="*/" --include="*.so" --exclude="*" ${CONDA_PREFIX}/lib/python3.12/site-packages/vllm/ vllm_backup/
    # there is also something fishy going on in vllm_flash_attn/ directory.
    # see their setup.py.
    # it seems they are manually copying this directory, so I should follow this too...
    rsync -av --prune-empty-dirs --include="*/" --include="*.py" --exclude="*" ${CONDA_PREFIX}/lib/python3.12/site-packages/vllm/vllm_flash_attn/ vllm_backup/vllm_flash_attn/

    uv pip install "vllm @ git+https://github.com/tdoublep/vllm@alora"

    rsync -av vllm_backup/ ${CONDA_PREFIX}/lib/python3.12/site-packages/vllm/
}

export -f install-vllm-fork

in-conda install-vllm-fork

