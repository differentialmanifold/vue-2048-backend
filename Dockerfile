FROM tensorflow/tensorflow:1.8.0-py3

COPY . /2048

WORKDIR /2048

RUN apt-get update && apt-get install -y --no-install-recommends \
        vim

RUN pip3 --no-cache-dir install \
        flask \
        flask-cors

# TensorBoard
EXPOSE 6006

EXPOSE 8888

CMD ["/bin/bash"]