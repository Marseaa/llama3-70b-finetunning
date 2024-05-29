
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y wget 

# instalando python
RUN apt-get install -y python3  && apt-get install -y python3-pip

WORKDIR /app

RUN pip install accelerate datasets
RUN pip install torch torchvision torchaudio transformers 

ENV NVIDIA_VISIBLE_DEVICES all

CMD ["python3", "data.py"]

