# Copyright (C) 2023-2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#
# Dockerfile - image with all Unified Runtime dependencies.
#

# Pull base image
FROM registry.hub.docker.com/library/ubuntu:22.04

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
	python3-pip"

# Unified Runtime's dependencies (installed via pip)
ARG UR_PYTHON_DEPS="\
	clang-format==15.0.7"

# Miscellaneous for our builds/CI (optional)
ARG MISC_DEPS="\
	clang \
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

# pip package is pinned to a version, but it's probably improperly parsed here
# hadolint ignore=DL3013
RUN pip3 install --no-cache-dir ${UR_PYTHON_DEPS}

# Install DPC++
COPY install_dpcpp.sh /opt/install_dpcpp.sh
ENV DPCPP_PATH=/opt/dpcpp
RUN /opt/install_dpcpp.sh

# Install libbacktrace
COPY install_libbacktrace.sh /opt/install_libbacktrace.sh
RUN /opt/install_libbacktrace.sh

# Add a new (non-root) 'test_user' and switch to it
ENV USER test_user
ENV USERPASS pass
RUN useradd -m "${USER}" -g sudo -p "$(mkpasswd ${USERPASS})"
USER test_user
