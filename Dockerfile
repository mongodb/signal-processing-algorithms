FROM quay.io/pypa/manylinux1_x86_64

RUN /opt/python/cp37-cp37m/bin/pip install poetry

ADD . signal_processing

WORKDIR signal_processing

ENTRYPOINT ["/signal_processing/scripts/deploy.sh"]
