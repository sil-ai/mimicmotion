FROM python:3.11

# Instalar dependencias del sistema necesarias INICIAL
#RUN apt-get update && apt-get install -y \
#    build-essential \
#    libgl1-mesa-glx \
#    ffmpeg \
#    libx264-dev \
#    libx265-dev \
#    libavcodec-extra \
#    libsm6 \
#    libxext6 \
#    && rm -rf /var/lib/apt/lists/*

RUN apt-get remove -y --purge ffmpeg

# docker build
RUN apt-get update && apt-get install -y\
    build-essential \
    ffmpeg \
    libavcodec-extra \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libpostproc-dev \
    libswresample-dev \
    libswscale-dev \
    yasm \
    pkg-config \
    libx264-dev \
    libx265-dev \
    libvpx-dev \
    libsm6 \
    libxext6 \
    libopus-dev \
    libmp3lame-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


#RUN apt-get install -y --no-install-recommends \
#    libfdk-aac-dev \

# Actualizar pip y configurar dependencias de Python
RUN pip install --upgrade pip setuptools wheel

RUN pip install --index-url https://download.pytorch.org/whl/cu117 \
    torch==2.0.1 \
    torchvision==0.15.2

RUN pip install \
    matplotlib \
    diffusers==0.27.0 \
    transformers==4.32.1 \
    decord==0.6.0 \
    einops \
    omegaconf \
    boto3==1.35.78 \
    botocore==1.35.99 \
    clearml==1.16.5 \
    PyAV \
    huggingface-hub==0.25.1 \
    imageio[ffmpeg] \
    accelerate

RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y av && \
    pip install --no-cache-dir av==12.0.0


RUN git clone https://github.com/sil-ai/micmicmotion.git



WORKDIR /Mimic-Motion

COPY . .



# RUN !git clone https://huggingface.co/vdo/stable-video-diffusion-img2vid-xt-1-1 MimicMotion/models/SVD/stable-video-diffusion-img2vid-xt-1-1