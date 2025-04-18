#!/bin/bash

set -e

USER_NAME=sycl

# By default Ubuntu sets an arbitrary UID value, that is different from host
# system. When CI passes default UID value of 1001, some of LLVM tools fail to
# discover user home directory and fail a few LIT tests. Fixes UID and GID to
# 1001, that is used as default by GitHub Actions.
USER_ID=1001

groupadd -g $USER_ID $USER_NAME && useradd $USER_NAME -u $USER_ID -g $USER_ID -m -s /bin/bash
# Add user to video/irc groups so that it can access GPU
usermod -aG video $USER_NAME
usermod -aG irc $USER_NAME

# group 109 is required for user to access PVC card.
groupadd -f -g 109 render
usermod -aG render $USER_NAME

# Allow user to run as sudo (without a password)
echo "$USER_NAME  ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
