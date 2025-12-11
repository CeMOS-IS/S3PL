FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY . /workspace

RUN apt update && apt install -q -y --no-install-recommends libglu1-mesa-dev libgomp1 libopenslide-dev python3 python3-pip python3-venv
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
