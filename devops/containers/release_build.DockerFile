FROM docker.io/aswf/ci-base:2025.1

ENV DEBIAN_FRONTEND=noninteractive

USER root

COPY scripts/create-sycl-user.sh /user-setup.sh
RUN /user-setup.sh

# Install ROCm (for RHEL 8.10), see:
# https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html
RUN dnf -y install https://repo.radeon.com/amdgpu-install/6.4.1/rhel/8.10/amdgpu-install-6.4.60401-1.el8.noarch.rpm
RUN dnf clean all
RUN wget https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
RUN rpm -e epel-release
RUN rpm -ivh epel-release-latest-8.noarch.rpm
RUN dnf install dnf-plugin-config-manager
RUN crb enable
RUN dnf -y install python3-setuptools python3-wheel
RUN usermod -a -G render,video sycl # Add the current user to the render and video groups
RUN dnf -y install rocm
RUN dnf list installed | grep -i rocm

COPY scripts/docker_entrypoint.sh /docker_entrypoint.sh

USER sycl

ENTRYPOINT ["/docker_entrypoint.sh"]