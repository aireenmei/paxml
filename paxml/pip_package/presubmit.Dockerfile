ARG image_name
ARG base_image="gcr.io/pax-on-cloud-project/${image_name}:latest"
FROM $base_image

RUN rm -rf /praxis && rm -rf /paxml/paxml && rm -rf /paxml/praxis
COPY . /paxml_new
RUN git clone https://github.com/google/praxis.git
RUN cd /praxis && git checkout d13f6d056dc1fef8858e7cc2d9eb572e6d9e3a7c
RUN mv /praxis/praxis /paxml/ && mv /paxml_new/paxml /paxml/

RUN cd /paxml && bazel build ...

WORKDIR /

CMD ["/bin/bash"]
