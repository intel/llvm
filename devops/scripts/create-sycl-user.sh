#!/bin/bash

# By default Ubuntu sets an arbitrary UID value, that is different from host
# system. When CI passes default UID value of 1001, some of LLVM tools fail to
# discover user home directory and fail a few LIT tests. Fixes UID and GID to
# 1001, that is used as default by GitHub Actions.
groupadd -g 1001 sycl && useradd sycl -u 1001 -g 1001 -m -s /bin/bash
# Add sycl user to video/irc groups so that it can access GPU
usermod -aG video sycl
usermod -aG irc sycl

# group 109 is required for sycl user to access PVC card.
groupadd -g 109 render
usermod -aG render sycl

if [[ -f /run/secrets/sycl_passwd ]]; then
  # When running in our CI environment, we restrict access to root.

  # Set password for sycl user
  cat /run/secrets/sycl_passwd | passwd -s sycl

  # Allow sycl user to run as sudo, but only with password
  echo "sycl  ALL=(root) PASSWD:ALL" >> /etc/sudoers
else
  # Otherwise, we allow password-less root to simplify building other
  # containers on top.

  # Allow sycl user to run as sudo passwrod-less
  echo "sycl  ALL=(root) NOPASSWD:ALL" >> /etc/sudoers
fi
