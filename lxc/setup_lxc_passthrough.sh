#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <CTID>"
    exit 1
fi

CTID="$1"
CONF="/etc/pve/lxc/${CTID}.conf"

if [[ ! -f "$CONF" ]]; then
    echo "Error: Container config ${CONF} does not exist."
    exit 1
fi

echo "[+] Setting up /dev/kfd and /dev/dri permissions on host..."
# Ensure render group ID permissions
RENDER_GID=$(getent group render | cut -d: -f3 || echo "105")
VIDEO_GID=$(getent group video | cut -d: -f3 || echo "44")

chmod 666 /dev/kfd
chmod 666 /dev/dri/renderD* || true

echo "[+] Verifying GPU presence via rocminfo or lspci..."
lspci -d 1002: | grep -E "Vega 20|MI50" || echo "[-] Warning: MI50 device not detected in lspci."

echo "[+] Ready. Append lxc/pve_lxc_mi50.conf entries to ${CONF} and restart container."
