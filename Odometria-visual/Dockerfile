FROM --platform=linux/arm64/v8 python:3.11-bookworm

WORKDIR /work

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libhdf5-103-1 \
    libhdf5-dev \
    libvtk9.1 \
    libgoogle-glog0v6 \
    libtesseract5 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libavcodec59 \
    libavformat59 \
    libavutil57 \
    libswscale6 \
    libswresample4 \
    && rm -rf /var/lib/apt/lists/*

RUN echo "/usr/lib/aarch64-linux-gnu/hdf5/serial" > /etc/ld.so.conf.d/hdf5.conf && \
    ldconfig

RUN pip install --upgrade pip

COPY opencv_contrib_python-4.11.0.86-cp311-cp311-linux_aarch64.whl /tmp/

RUN pip install "numpy==1.26.4"

RUN pip install /tmp/opencv_contrib_python-4.11.0.86-cp311-cp311-linux_aarch64.whl

RUN pip install \
    ipykernel \
    jupyter \
    notebook \
    jupyterlab \
    "matplotlib<3.9" \
    "pandas<3" \
    "scipy<1.14" \
    pillow

RUN pip install --force-reinstall "numpy==1.26.4"

RUN python -m ipykernel install \
    --name odometria-arm64 \
    --display-name "Python 3.11 ARM64 - Odometria"

CMD ["sleep", "infinity"]