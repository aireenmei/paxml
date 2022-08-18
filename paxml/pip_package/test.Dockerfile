ARG cpu_base_image="ubuntu:20.04"
ARG base_image=$cpu_base_image
FROM $base_image

RUN ls && find . -name "paxml"

WORKDIR "/"

CMD ["/bin/bash"]
