# Base image with Python 3.10, PyTorch 2.2, CUDA 11.8, Ubuntu 22.04
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

# Install apt packages
RUN apt-get update && \
    apt-get install -y ffmpeg build-essential python3-dev libcairo2-dev pkg-config python3-pip \
    libdbus-1-dev libgirepository1.0-dev git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install pip packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    numpy==2.2.6 \
    scipy==1.15.3 \
    pandas==2.3.0 \
    matplotlib==3.10.3 \
    scikit-learn==1.7.0 \
    tqdm==4.67.1 \
    addict==2.4.0 \
    antlr4-python3-runtime==4.9.3 \
    anyio==4.0.0 \
    argon2-cffi==23.1.0 \
    argon2-cffi-bindings==21.2.0 \
    arrow==1.3.0 \
    asttokens==2.4.1 \
    async-lru==2.0.4 \
    attrs==23.1.0 \
    Babel==2.13.1 \
    beautifulsoup4==4.12.2 \
    bleach==6.1.0 \
    blinker==1.9.0 \
    certifi==2022.12.7 \
    cffi==1.16.0 \
    charset-normalizer==2.1.1 \
    click==8.2.1 \
    comm==0.2.0 \
    ConfigArgParse==1.7.1 \
    contourpy==1.3.2 \
    cryptography==3.4.8 \
    cycler==0.12.1 \
    dash==3.0.4 \
    dbus-python==1.2.18 \
    debugpy==1.8.0 \
    decorator==5.1.1 \
    defusedxml==0.7.1 \
    distro==1.7.0 \
    docstring_parser==0.16 \
    einops==0.8.1 \
    entrypoints==0.4 \
    exceptiongroup==1.1.3 \
    executing==2.0.1 \
    fastjsonschema==2.18.1 \
    ffmpeg-python==0.2.0 \
    filelock==3.9.0 \
    Flask==3.0.3 \
    fonttools==4.58.2 \
    fqdn==1.5.1 \
    fsspec==2025.5.1 \
    ftfy==6.3.1 \
    future==1.0.0 \
    h5py==3.14.0 \
    hf-xet==1.1.3 \
    httplib2==0.20.2 \
    huggingface-hub==0.32.5 \
    hydra-core==1.3.2 \
    idna==3.4 \
    importlib-metadata==4.6.4 \
    ipykernel==6.26.0 \
    ipython==8.17.2 \
    ipython-genutils==0.2.0 \
    ipywidgets==8.1.1 \
    isoduration==20.11.0 \
    itsdangerous==2.2.0 \
    jedi==0.19.1 \
    jeepney==0.7.1 \
    Jinja2==3.1.2 \
    joblib==1.5.1 \
    json5==0.9.14 \
    jsonpointer==2.4 \
    jsonschema==4.19.2 \
    jsonschema-specifications==2023.7.1 \
    jupyter-archive==3.4.0 \
    jupyter-contrib-core==0.4.2 \
    jupyter-contrib-nbextensions==0.7.0 \
    jupyter-events==0.9.0 \
    jupyter-highlight-selected-word==0.2.0 \
    jupyter-lsp==2.2.0 \
    jupyter-nbextensions-configurator==0.6.3 \
    jupyter_client==7.4.9 \
    jupyter_core==5.5.0 \
    jupyter_server==2.10.0 \
    jupyter_server_terminals==0.4.4 \
    jupyterlab==4.0.8 \
    jupyterlab-pygments==0.2.2 \
    jupyterlab-widgets==3.0.9 \
    jupyterlab_server==2.25.0 \
    keyring==23.5.0 \
    kiwisolver==1.4.8 \
    launchpadlib==1.10.16 \
    lazr.restfulclient==0.14.4 \
    lazr.uri==1.0.6 \
    lxml==4.9.3 \
    markdown-it-py==3.0.0 \
    MarkupSafe==2.1.2 \
    matplotlib-inline==0.1.6 \
    mdurl==0.1.2 \
    mistune==3.0.2 \
    more-itertools==8.10.0 \
    mpmath==1.3.0 \
    narwhals==1.42.0 \
    natsort==8.4.0 \
    nbclassic==1.0.0 \
    nbclient==0.9.0 \
    nbconvert==7.11.0 \
    nbformat==5.9.2 \
    nest-asyncio==1.5.8 \
    networkx==3.0 \
    ninja==1.11.1.4 \
    notebook==6.5.5 \
    notebook_shim==0.2.3 \
    oauthlib==3.2.0 \
    omegaconf==2.3.0 \
    open3d==0.19.0 \
    opencv-python==4.11.0.86 \
    orjson==3.10.18 \
    overrides==7.4.0 \
    packaging==23.2 \
    packaging \
    pandocfilters==1.5.0 \
    parso==0.8.3 \
    pexpect==4.8.0 \
    Pillow==9.3.0 \
    platformdirs==3.11.0 \
    plotly==6.1.2 \
    prometheus-client==0.18.0 \
    prompt-toolkit==3.0.39 \
    psutil==5.9.6 \
    ptyprocess==0.7.0 \
    pure-eval==0.2.2 \
    pyarrow==20.0.0 \
    pycparser==2.21 \
    PyGments==2.16.1 \
    PyGObject==3.42.1 \
    PyJWT==2.3.0 \
    pyliblzfse==0.4.1 \
    pyparsing==2.4.7 \
    pypng==0.20220715.0 \
    pyquaternion==0.9.9 \
    python-dateutil==2.8.2 \
    python-json-logger==2.0.7 \
    pytz==2025.2 \
    PyYAML==6.0.1 \
    pyzmq==24.0.1 \
    referencing==0.30.2 \
    regex==2024.11.6 \
    requests==2.31.0 \
    rerun-sdk==0.23.3 \
    retrying==1.3.4 \
    rfc3339-validator==0.1.4 \
    rfc3986-validator==0.1.1 \
    rich==14.0.0 \
    rpds-py==0.12.0 \
    safetensors==0.5.3 \
    SecretStorage==3.3.1 \
    Send2Trash==1.8.2 \
    shtab==1.7.2 \
    six==1.16.0 \
    sniffio==1.3.0 \
    soupsieve==2.5 \
    stack-data==0.6.3 \
    sympy==1.13.3 \
    terminado==0.17.1 \
    threadpoolctl==3.6.0 \
    timm==1.0.15 \
    tinycss2==1.2.1 \
    tomli==2.0.1 \
    tornado==6.3.3 \
    traitlets==5.13.0 \
    triton==3.3.1 \
    typeguard==4.4.3 \
    types-python-dateutil==2.8.19.14 \
    typing_extensions==4.14.0 \
    tyro==0.9.24 \
    tzdata==2025.2 \
    uri-template==1.3.0 \
    urllib3==1.26.13 \
    wadllib==1.3.6 \
    wcwidth==0.2.9 \
    webcolors==1.13 \
    webencodings==0.5.1 \
    websocket-client==1.6.4 \
    Werkzeug==3.0.6 \
    widgetsnbextension==4.0.9 \
    zipp==1.0.0

RUN pip install git+https://github.com/openai/CLIP.git

# ssh
RUN     mkdir -p ~/.ssh && \
        ssh-keyscan github.com >> ~/.ssh/known_hosts

# aria
RUN pip install projectaria_tools

# depth pro
RUN pip install git+https://github.com/apple/ml-depth-pro.git

# install hamer
RUN --mount=type=ssh,id=github pip install git+ssh://git@github.com/3dvision-cecn/hamer.git 
RUN pip install git+https://github.com/ViTAE-Transformer/ViTPose.git

# install avion
RUN --mount=type=ssh,id=github pip install git+ssh://git@github.com/3dvision-cecn/AVION_BUT.git

# install edgetam
RUN pip install git+https://github.com/facebookresearch/EdgeTAM.git

RUN pip install webdataset ultralytics transformers==4.51.3 torch_geometric==2.6.1

# renderer
RUN apt-get update && apt-get install -y libglfw3-dev libgles2-mesa-dev openssh-server

# openclip
RUN pip install git+https://github.com/mlfoundations/open_clip.git

# apex
RUN pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
RUN CUDA_HOME=/usr/local/cuda-12.4/ pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git
# flash attn
RUN CUDA_HOME=/usr/local/cuda-12.4/ MAX_JOBS=8 pip install flash-attn --no-build-isolation

# omni ogn
RUN --mount=type=ssh,id=github git clone git@github.com:3dvision-cecn/OMNI-DC.git /tmp/omni_dc
RUN --mount=type=ssh,id=github pip install -e /tmp/omni_dc

RUN pip install mmsegmentation==0.22.1 mmcv==1.4.4

# ----  SSH support  -------------------------------------------------
# Install OpenSSH, create runtime dirs and a non-root user
RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends openssh-server tmux && \
    mkdir -p /var/run/sshd && \
    useradd -m -s /bin/bash runpod && \
    echo "runpod ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Generate host keys at build time (optional but speeds first boot)
RUN ssh-keygen -A

# Copy *public* key or let RunPod inject it with an env-var at start-time
ARG SSH_PUB_KEY="ssh-ed25519 AAAA... user@example"
RUN install -d -m 700 /home/runpod/.ssh && \
    echo "${SSH_PUB_KEY}" > /home/runpod/.ssh/authorized_keys && \
    chmod 600 /home/runpod/.ssh/authorized_keys && \
    chown -R runpod:runpod /home/runpod/.ssh

# Listen on the classic port in the container
EXPOSE 22

# Run the daemon in the foreground so Docker keeps the container alive
CMD ["/bin/bash", "-c", "/usr/sbin/sshd -D & sleep infinity"]


