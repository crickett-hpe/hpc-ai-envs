#!/bin/bash
# Build PMIx to match host Slurm PMIx v5
# This allows container OpenMPI to work with host Slurm PMIx

set -ex

# Install dependencies that may not be in base image
apt-get update && apt-get install -y --no-install-recommends \
    libevent-dev \
    libhwloc-dev \
    || true

PMIX_VERSION=${PMIX_VERSION:-5.0.3}
PMIX_SRC_DIR=/tmp/pmix-src
PMIX_URL="https://github.com/openpmix/openpmix/releases/download/v${PMIX_VERSION}/pmix-${PMIX_VERSION}.tar.gz"

echo "=== Building PMIx ${PMIX_VERSION} ==="

# Debug: show what hwloc and libevent files exist
echo "=== Checking hwloc installation ==="
dpkg -L libhwloc-dev 2>/dev/null | grep -E "\.so|\.a|\.pc" | head -10 || true
ls -la /usr/lib/*/libhwloc* 2>/dev/null | head -10 || true
echo "=== Checking libevent installation ==="
dpkg -L libevent-dev 2>/dev/null | grep -E "\.so|\.a" | head -10 || true
ls -la /usr/lib/*/libevent* 2>/dev/null | head -10 || true

# Find the multiarch library directory
MULTIARCH_DIR=$(dpkg-architecture -qDEB_HOST_MULTIARCH 2>/dev/null || echo "")
if [ -n "$MULTIARCH_DIR" ] && [ -d "/usr/lib/$MULTIARCH_DIR" ]; then
    LIBDIR="/usr/lib/$MULTIARCH_DIR"
else
    LIBDIR="/usr/lib"
fi
echo "Using LIBDIR=$LIBDIR"

mkdir -p ${PMIX_SRC_DIR}
cd ${PMIX_SRC_DIR}

wget ${PMIX_URL}
tar -xzf pmix-${PMIX_VERSION}.tar.gz
cd pmix-${PMIX_VERSION}

# Configure PMIx
# - Install to HPC_DIR so OpenMPI can find it
# - Disable munge (not available in container, host handles security via Slurm)
# - Use pkg-config to find hwloc and libevent (handles multiarch paths)
export PKG_CONFIG_PATH="${LIBDIR}/pkgconfig:/usr/lib/pkgconfig:${PKG_CONFIG_PATH}"
export LDFLAGS="-L${LIBDIR}"

./configure \
    --prefix=${HPC_DIR} \
    --disable-munge \
    --disable-psec-dummy-handshake \
    --enable-shared \
    --disable-static

make -j$(nproc)
make install

# Create default MCA parameters file to avoid problematic components at runtime
mkdir -p ${HPC_DIR}/etc
cat > ${HPC_DIR}/etc/pmix-mca-params.conf << 'EOF'
# PMIx MCA defaults for container compatibility with host Slurm
# Disable munge (host Slurm handles security)
pmix_psec = ^munge
# Use hash GDS instead of shmem2 (more container-friendly)
pmix_gds = hash
# Disable usock PTL
pmix_ptl = ^usock
EOF

cd /tmp
rm -rf ${PMIX_SRC_DIR}

echo "=== PMIx ${PMIX_VERSION} installed to ${HPC_DIR} ==="
