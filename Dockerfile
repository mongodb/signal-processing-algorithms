FROM quay.io/pypa/manylinux2014_x86_64

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN /opt/python/cp38-cp38/bin/pip install poetry

ADD . signal_processing

WORKDIR signal_processing

ENTRYPOINT ["/signal_processing/scripts/deploy.sh"]
