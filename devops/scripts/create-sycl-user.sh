#!/bin/bash

set -e

if [[ $# -eq 0 ]]; then
  # When launched without arguments, we assume that it was launched as part of
  # CI workflow and therefore a different kind of user is created
  USER_NAME=sycl_ci
  SET_PASSWD=true

  # By default Ubuntu sets an arbitrary UID value, that is different from host
  # system. When CI passes default UID value of 1001, some of LLVM tools fail to
  # discover user home directory and fail a few LIT tests. Fixes UID and GID to
  # 1001, that is used as default by GitHub Actions.
  USER_ID=1001
else
  if [[ "${1:-}" != "--regular" ]]; then
    echo "The only supported argument is --regular!"
    exit 1
  fi
  USER_NAME=sycl
  SET_PASSWD=false

  # Some user id which is different from the one assigned to sycl_ci user
  USER_ID=1234
fi

groupadd -g $USER_ID $USER_NAME && useradd $USER_NAME -u $USER_ID -g $USER_ID -m -s /bin/bash
# Add user to video/irc groups so that it can access GPU
usermod -aG video $USER_NAME
usermod -aG irc $USER_NAME

# group 109 is required for user to access PVC card.
groupadd -f -g 109 render
usermod -aG render $USER_NAME

if [[ $SET_PASSWD == true ]]; then
  if [[ ! -f /run/secrets/sycl_ci_passwd ]]; then
    echo "Password is requested, but /run/secrets/sycl_ci_passwd doesn't exist!"
    exit 2
  fi

  # Set password for user
  echo "$USER_NAME:$(cat /run/secrets/sycl_ci_passwd)" | chpasswd

  # Allow user to run as sudo, but only with password
  echo "$USER_NAME  ALL=(ALL) PASSWD:ALL" >> /etc/sudoers
else
  echo "$USER_NAME  ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
fi
