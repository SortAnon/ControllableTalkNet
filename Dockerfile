FROM nvidia/cuda:11.0.3-base-ubuntu20.04 
ENV TZ=Etc/GMT
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y git ffmpeg python3.8 python3-pip curl
RUN git clone -q https://github.com/SortAnon/ControllableTalkNet /talknet
RUN git clone -q https://github.com/SortAnon/hifi-gan /talknet/hifi-gan
RUN python3.8 -m pip install pandas==1.1
RUN python3.8 -m pip install cachetools==4.2.4
RUN python3.8 -m pip install importlib-metadata==4.13.0
RUN python3.8 -m pip install PyYAML==5.4.1
RUN python3.8 -m pip install pydantic==1.9.2
RUN python3.8 -m pip install rich==12.0.1
RUN python3.8 -m pip install hmmlearn==0.2.5
RUN python3.8 -m pip --no-cache-dir install -r "/talknet/requirements.txt" -f https://download.pytorch.org/whl/torch_stable.html
RUN python3.8 -m pip --no-cache-dir install git+https://github.com/SortAnon/NeMo.git
RUN python3.8 -m pip uninstall -y pesq
RUN python3.8 -m pip install pesq==0.0.2
RUN python3.8 -m pip install werkzeug==2.0.3
RUN touch /talknet/is_docker
WORKDIR /talknet
EXPOSE 8050

RUN printf "#!/bin/bash \
    \necho Updating TalkNet... \
    \ngit -C /talknet reset --hard origin/main -q \
    \ngit -C /talknet pull origin main -q \
    \necho Updating HiFi-GAN... \
    \ngit -C /talknet/hifi-gan reset --hard origin/master -q \
    \ngit -C /talknet/hifi-gan pull origin master -q \
    \necho Updating Python dependencies... \
    \npython3.8 -m pip --quiet --no-cache-dir install -r /talknet/requirements.txt -f https://download.pytorch.org/whl/torch_stable.html \
    \npython3.8 -m pip --quiet --no-cache-dir install git+https://github.com/SortAnon/NeMo.git \
    \necho Launching TalkNet... \
    \npython3.8 talknet_offline.py\n" > /talknet/docker_launch.sh

RUN chmod +x /talknet/docker_launch.sh
CMD ["./docker_launch.sh"]
