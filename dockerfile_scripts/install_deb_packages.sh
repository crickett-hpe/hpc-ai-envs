#!/usr/bin/env bash

mkdir -p /var/run/sshd

apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    autotools-dev \
    build-essential \
    ca-certificates \
    curl \
    daemontools \
    debhelper \
    devscripts \
    ibverbs-providers \
    libibverbs1 \
    libkrb5-dev \
    librdmacm1 \
    libssl-dev \
    libtool \
    git \
    krb5-user \
    g++ \
    cmake \
    make \
    openssh-client \
    openssh-server \
    pkg-config \
    wget \
    nfs-common \
    libnuma1 \
    libnuma-dev \
    libpmi2-0-dev \
    hwloc \
    libhwloc-dev \
    libjson-c-dev \
    libnl-3-dev \
    libnl-*-3-dev \
    libconfig-dev \
    libuv1-dev \
    fuse \
    libfuse-dev \
    libyaml-dev \
    libsensors-dev \
    gdb \
    unattended-upgrades \
  && unattended-upgrade \
  && rm -rf /var/lib/apt/lists/* \
  && rm /etc/ssh/ssh_host_ecdsa_key \
  && rm /etc/ssh/ssh_host_ed25519_key \
  && rm /etc/ssh/ssh_host_rsa_key
