# Copyright (C) 2023-2024 Intel Corporation
# Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE.TXT
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#
# Dockerfile - a 'recipe' for Docker to build an image of fedora-based
#              environment for building the Unified Runtime project.
#

# Pull base image ("40")
FROM registry.hub.docker.com/library/fedora@sha256:5ce8497aeea599bf6b54ab3979133923d82aaa4f6ca5ced1812611b197c79eb0

# Set environment variables
ENV OS fedora
ENV OS_VER 40
ENV NOTTY 1

# Additional parameters to build docker without building components.
# These ARGs can be set in docker building phase and are used
# within bash scripts (executed within docker).
ARG SKIP_DPCPP_BUILD
ARG SKIP_LIBBACKTRACE_BUILD

# Base development packages
ARG BASE_DEPS="\
	cmake \
	git \
	make"

# Unified Runtime's dependencies
ARG UR_DEPS="\
	doxygen \
	python3 \
	python3-pip"

# Miscellaneous for our builds/CI (optional)
ARG MISC_DEPS="\
	clang \
	ncurses-libs-6.4 \
	passwd \
	sudo \
	wget"

# Update and install required packages
RUN dnf update -y \
 && dnf install -y \
	${BASE_DEPS} \
	${UR_DEPS} \
	${MISC_DEPS} \
 && dnf clean all

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

# Add a new (non-root) 'test_user'
ENV USER test_user
ENV USERPASS pass
# Change shell to bash with safe pipe usage
SHELL [ "/bin/bash", "-o", "pipefail", "-c" ]
RUN useradd -m ${USER} \
 && echo "${USER}:${USERPASS}" | chpasswd \
 && gpasswd wheel -a ${USER}

# Change shell back to default and switch to 'test_user'
SHELL ["/bin/sh", "-c"]
USER test_user
