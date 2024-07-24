# Copyright (C) 2023-2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#
# Dockerfile - a 'recipe' for Docker to build an image of opensuse-leap-based
#              environment for building the Unified Runtime project.
#

# Pull base image ("15")
FROM registry.hub.docker.com/opensuse/leap@sha256:1cf79e78bb69f39fb2f78a7c2c7ebc4b64cf8d82eb1df76cd36767a595ada7a8

# Set environment variables
ENV OS opensuse-leap
ENV OS_VER 15
ENV NOTTY 1

# Additional parameters to build docker without building components.
# These ARGs can be set in docker building phase and are used
# within bash scripts (executed within docker).
ARG SKIP_DPCPP_BUILD
ARG SKIP_LIBBACKTRACE_BUILD

# Base development packages
ARG BASE_DEPS="\
	cmake \
	gcc \
	gcc-c++ \
	git \
	glibc-devel \
	libstdc++-devel \
	make"

# Unified Runtime's dependencies
ARG UR_DEPS="\
	doxygen \
	python3 \
	python3-devel \
	python3-pip"

# Miscellaneous for our builds/CI (optional)
ARG MISC_DEPS="\
	clang \
	gzip \
	libncurses5 \
	sudo \
	tar \
	wget"

# add openSUSE Leap 15.5 Oss repo
RUN zypper ar -f https://download.opensuse.org/distribution/leap/15.5/repo/oss/ oss

# Update and install required packages
RUN zypper update -y \
 && zypper install -y \
	${BASE_DEPS} \
	${UR_DEPS} \
	${MISC_DEPS} \
 && zypper clean all

# Prepare a dir (accessible by anyone)
RUN mkdir --mode 777 /opt/ur/

# Additional dev. dependencies (installed via pip)
#
# It's actively used and tested only on selected distros. Be aware
# they may not work, because pip packages list differ from OS to OS.
COPY third_party/requirements.txt /opt/ur/requirements.txt

# Install DPC++
COPY .github/docker/install_dpcpp.sh /opt/ur/install_dpcpp.sh
ENV DPCPP_PATH=/opt/dpcpp
RUN /opt/ur/install_dpcpp.sh

# Install libbacktrace
COPY .github/docker/install_libbacktrace.sh /opt/ur/install_libbacktrace.sh
RUN /opt/ur/install_libbacktrace.sh

# Add a new (non-root) 'test_user' and switch to it
ENV USER test_user
ENV USERPASS pass
ENV PFILE ./password
RUN useradd -m ${USER} \
 && echo ${USERPASS} > ${PFILE} \
 && echo ${USERPASS} >> ${PFILE} \
 && passwd ${USER} < ${PFILE} \
 && rm -f ${PFILE} \
 && sed -i 's/# %wheel/%wheel/g' /etc/sudoers \
 && groupadd wheel \
 && gpasswd wheel -a ${USER}
USER test_user
