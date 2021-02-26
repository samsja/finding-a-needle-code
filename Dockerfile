
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

USER $NB_UID


RUN apt-get update

RUN python -m pip install -U pip 

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN rm requirements.txt

RUN pip install jupyter tensorboard

# jupyter

workdir /workspace
EXPOSE 8888 

CMD ["bash", "-c", "jupyter notebook --notebook-dir=/workspace --ip 0.0.0.0 --no-browser --allow-root"]
