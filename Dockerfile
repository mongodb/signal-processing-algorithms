FROM quay.io/pypa/manylinux2010_x86_64

RUN /opt/python/cp38-cp38/bin/pip install poetry

ADD . signal_processing

WORKDIR signal_processing

ENTRYPOINT ["/signal_processing/scripts/deploy.sh"]
