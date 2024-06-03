
# check se essa versão do cuda é compatível com os pacotes.
#FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# melhor usar algo como:
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1 \
    CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST="8.6"
RUN rm /bin/sh && ln -s /bin/bash /bin/sh


ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y wget 

# instalando python
RUN apt-get install -y python3  && apt-get install -y python3-pip

RUN pip install accelerate datasets
RUN pip install torch torchvision torchaudio transformers 

ENV NVIDIA_VISIBLE_DEVICES all
WORKDIR /app
CMD ["python3", "data.py"]

