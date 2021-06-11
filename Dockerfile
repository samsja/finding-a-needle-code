FROM nvcr.io/nvidia/pytorch:21.05-py3
# Create a working directory
RUN mkdir /app
WORKDIR /app

RUN conda install jupyter && conda clean -ya 
RUN conda install -c conda-forge jupyterlab && conda clean -ya 


RUN conda install -c conda-forge jupyterlab_code_formatter
RUN conda install -c conda-forge xeus-python 

RUN conda install -c anaconda pip 
RUN pip install tb-nightly
RUN pip install black isort


COPY requirements.txt .
RUN conda install -c conda-forge  --file requirements.txt

EXPOSE 8888 

CMD ["bash", "-c", "jupyter notebook --notebook-dir=/workspace --ip 0.0.0.0 --no-browser --allow-root"]
