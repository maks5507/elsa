FROM continuumio/anaconda3 AS build

ARG renderer_queue
ARG demo_queue
ARG rmq_connect

RUN apt update; apt install vim procps build-essential git -y;

RUN git clone https://github.com/maks5507/amqp-interface.git; cd amqp-interface; python setup.py build; pip install .;
RUN git clone https://github.com/maks5507/elsa.git; cd elsa; python setup.py build; pip install .;

RUN export OMP_NUM_THREADS=1

COPY . /root/

RUN pip install -r /root/requirements.txt
RUN python /root/setup/init_amqp.py -c $rmq_connect -rq $renderer_queue -dq $demo_queue
