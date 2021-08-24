FROM nvidia/cuda:11.0.3-base-ubuntu20.04 
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y git ffmpeg python3.8 python3-pip curl
RUN git clone -q https://github.com/SortAnon/ControllableTalkNet /talknet
RUN git clone -q https://github.com/SortAnon/hifi-gan /talknet/hifi-gan
RUN python3.8 -m pip --no-cache-dir install -r "/talknet/requirements.txt" -f https://download.pytorch.org/whl/torch_stable.html
RUN python3.8 -m pip --no-cache-dir install git+https://github.com/SortAnon/NeMo.git
RUN python3.8 -m pip uninstall -y pesq
RUN python3.8 -m pip install pesq==0.0.2
RUN touch /talknet/is_docker
WORKDIR /talknet
EXPOSE 8050

RUN printf "#!/bin/bash \
    \necho Updating TalkNet... \
    \ngit -C /talknet reset --hard origin/main -q \
    \ngit -C /talknet pull origin main -q \
    \necho Updating HiFi-GAN... \
    \ngit -C /talknet reset --hard origin/main -q \
    \ngit -C /talknet pull origin main -q \
    \necho Updating Python dependencies... \
    \npython3.8 -m pip --quiet --no-cache-dir install -r /talknet/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html \
    \npython3.8 -m pip --quiet --no-cache-dir install git+https://github.com/SortAnon/NeMo.git \
    \necho Launching TalkNet... \
    \npython3.8 talknet_offline.py\n" > /talknet/docker_launch.sh

RUN chmod +x /talknet/docker_launch.sh
CMD ["./docker_launch.sh"]
