FROM docker.io/nvidia/cuda:12.1.0-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

USER root

# CUDA apt key rotation fix for older base images.
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Install a recent enough CMake from Kitware's repository.
RUN apt-get update -qq && \
		apt-get install --no-install-recommends -yqq ca-certificates gpg wget && \
		wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
			gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
		echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main" | \
			tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
		apt-get update -qq && \
		apt-get install --no-install-recommends -yqq cmake

# Ubuntu 18.04 ships with Python 3.6, but buildbot scripts need at least 3.8.
RUN apt-get update -qq && \
		apt-get install --no-install-recommends -yqq software-properties-common && \
		add-apt-repository -y ppa:deadsnakes/ppa && \
		apt-get update -qq && \
		apt-get install --no-install-recommends -yqq \
			python3.8 \
			python3.8-dev \
			python3.8-venv \
			python3.8-distutils \
			python3-pip && \
		update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 10 && \
		python3 --version

# Install dependencies required to configure and build intel/llvm with
# buildbot/configure.py and buildbot/compile.py.
RUN apt-get update -qq && \
		apt-get install --no-install-recommends -yqq \
			git \
			build-essential \
			ccache \
			python3-setuptools \
			python3-psutil \
			pkg-config \
			ocl-icd-opencl-dev \
			libffi-dev \
			libxml2-dev \
			libedit-dev \
			libncurses5-dev \
			unzip \
			zip \
			ninja-build && \
		rm -rf /var/lib/apt/lists/*

# Build libhwloc 2.3.0+ from sources (Ubuntu 18.04 ships with 1.11.6).
RUN cd /tmp && \
		wget -q https://www.open-mpi.org/software/hwloc/v2.3/downloads/hwloc-2.3.0.tar.gz && \
		tar xzf hwloc-2.3.0.tar.gz && \
		cd hwloc-2.3.0 && \
		./configure --prefix=/usr/local --disable-static && \
		make -j$(nproc) && \
		make install && \
		ldconfig && \
		cd /tmp && \
		rm -rf hwloc-2.3.0*

# Build zstd 1.5.7 from sources with -fPIC flag (Ubuntu 18.04 ships with 1.3.3).
RUN cd /tmp && \
		wget -q https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz && \
		tar xzf zstd-1.5.7.tar.gz && \
		cd zstd-1.5.7 && \
		CFLAGS="-fPIC" CXXFLAGS="-fPIC" make -j$(nproc) && \
		make install PREFIX=/usr/local && \
		cp /usr/local/include/zstd.h /usr/include/ && \
		cp /usr/local/lib/libzstd.* /usr/lib/ 2>/dev/null || true && \
		ldconfig && \
		cd /tmp && \
		rm -rf zstd-1.5.7*

COPY scripts/create-sycl-user.sh /user-setup.sh
RUN /user-setup.sh

COPY scripts/docker_entrypoint.sh /docker_entrypoint.sh

COPY actions/cached_checkout /actions/cached_checkout
COPY actions/cleanup /actions/cleanup

USER sycl

ENTRYPOINT ["/docker_entrypoint.sh"]
