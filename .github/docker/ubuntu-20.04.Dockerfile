# Copyright (C) 2023-2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#
# Dockerfile - a 'recipe' for Docker to build an image of ubuntu-based
#              environment for building the Unified Runtime project.
#

# Pull base image ("20.04")
FROM registry.hub.docker.com/library/ubuntu@sha256:d86db849e59626d94f768c679aba441163c996caf7a3426f44924d0239ffe03f

# Set environment variables
ENV OS ubuntu
ENV OS_VER 20.04
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
	git"

# Unified Runtime's dependencies
ARG UR_DEPS="\
	doxygen \
	python3 \
	python3-pip"

# Miscellaneous for our builds/CI (optional)
ARG MISC_DEPS="\
	clang \
	g++-7 \
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

# Install CMake from source (the version in apt it's too old)
RUN wget https://cmake.org/files/v3.20/cmake-3.20.0-linux-x86_64.sh -O cmake.sh \
 && chmod +x cmake.sh \
 && ./cmake.sh --skip-license --prefix=/usr/local

# Prepare a dir (accessible by anyone)
RUN mkdir --mode 777 /opt/ur/

# Additional dev. dependencies (installed via pip)
COPY third_party/requirements.txt /opt/ur/requirements.txt
RUN pip3 install --no-cache-dir -r /opt/ur/requirements.txt

# Install libbacktrace
COPY .github/docker/install_libbacktrace.sh /opt/ur/install_libbacktrace.sh
RUN /opt/ur/install_libbacktrace.sh

# Add a new (non-root) 'test_user' and switch to it
ENV USER test_user
ENV USERPASS pass
RUN useradd -m "${USER}" -g sudo -p "$(mkpasswd ${USERPASS})"
USER test_user
