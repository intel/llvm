# Copyright (C) 2023-2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#
# Dockerfile - a 'recipe' for Docker to build an image of ubuntu-based
#              environment for building the Unified Runtime project.
#

# Pull base image ("22.04")
FROM registry.hub.docker.com/library/ubuntu@sha256:0eb0f877e1c869a300c442c41120e778db7161419244ee5cbc6fa5f134e74736

# Set environment variables
ENV OS ubuntu
ENV OS_VER 22.04
ENV NOTTY 1
ENV DEBIAN_FRONTEND noninteractive

# Additional parameters to build docker without building components.
# These ARGs can be set in docker building phase and are used
# within bash scripts (executed within docker).
ARG SKIP_DPCPP_BUILD
ARG SKIP_LIBBACKTRACE_BUILD

# Base development packages
ARG BASE_DEPS="\
	build-essential \
	cmake \
	git"

# Unified Runtime's dependencies
ARG UR_DEPS="\
	doxygen \
	python3 \
	python3-pip \
	libhwloc-dev"

# Miscellaneous for our builds/CI (optional)
ARG MISC_DEPS="\
	clang \
	libncurses5 \
	sudo \
	wget \
	whois"

# Update and install required packages
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
	${BASE_DEPS} \
	${UR_DEPS} \
	${MISC_DEPS} \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean all

# Prepare a dir (accessible by anyone)
RUN mkdir --mode 777 /opt/ur/

# Additional dev. dependencies (installed via pip)
COPY third_party/requirements.txt /opt/ur/requirements.txt
RUN pip3 install --no-cache-dir -r /opt/ur/requirements.txt

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
RUN useradd -m "${USER}" -g sudo -p "$(mkpasswd ${USERPASS})"
USER test_user
