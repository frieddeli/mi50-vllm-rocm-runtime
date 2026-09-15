# AMD Instinct MI50 vLLM ROCm Runtime & Proxmox LXC Suite

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Architecture: gfx906](https://img.shields.io/badge/Architecture-gfx906%20(Vega%2020)-purple.svg)]()
[![Hardware: 32GB HBM2](https://img.shields.io/badge/Hardware-32GB%20HBM2%20%40%201024%20GB%2Fs-red.svg)]()
[![vLLM: Continuous Batching](https://img.shields.io/badge/vLLM-Continuous%20Batching-green.svg)]()
[![Engineering Log](https://img.shields.io/badge/Engineering%20Log-Portfolio%20Deep%20Dive-blueviolet.svg)](https://frieddeli.github.io/Portfolio-Website/#/blog/b4)

High-throughput continuous-batching inference runtime for the retired **AMD Instinct MI50** (32GB HBM2, `gfx906`, Vega 20) under modern ROCm in unprivileged Proxmox VE LXC containers.

> [!NOTE]
> **Companion Engineering Log & Architecture Deep Dive:**  
> A detailed technical writeup detailing the LXC cgroup passthrough mechanics, rocBLAS GEMM kernel compilation, and continuous batching benchmarks is documented on my personal portfolio blog:
> - [Build Log MSN-013: Deploying vLLM on AMD Instinct MI50: 422 tok/s on Retired Enterprise Silicon](https://frieddeli.github.io/Portfolio-Website/#/blog/b4)

---

## 1. Motivation & Hardware Capabilities

The **AMD Instinct MI50** offers exceptional value-per-gigabyte for self-hosted LLM inference, packing **32 GB of high-bandwidth memory across a 4,096-bit HBM2 bus delivering 1,024 GB/s of raw bandwidth** alongside 60 Compute Units (3,840 stream processors).

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│ AMD INSTINCT MI50 SILICON ARCHITECTURE OVERVIEW                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│ • Architecture:      Vega 20 / GCN 5.1 (Target ISA: gfx906)                      │
│ • Compute Units:     60 CUs (3,840 Stream Processors)                            │
│ • Peak FP16 Compute: 26.8 TFLOPS (FP16 half-precision)                           │
│ • VRAM Capacity:     32 GB HBM2 (4 stacks @ 1,000 MHz)                           │
│ • Memory Bus Width:  4,096-bit Ultra-Wide Interface                              │
│ • Memory Bandwidth:  1,024 GB/s (1.024 TB/s)                                     │
│ • Host Interconnect: PCIe 4.0 x16 (31.5 GB/s bidirectional)                      │
│ • Form Factor:       Passive Server Heatsink (Requires Forced-Air Induction)     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### The Software Challenge

AMD officially retired the Vega 20 (`gfx906`) architecture from mainstream ROCm releases after version 5.7. Running modern high-throughput inference engines like vLLM under modern Linux kernels introduces distinct engineering obstacles:

1. **Virtualization Overhead:** Traditional GPU virtualization requires full KVM virtual machines with dedicated PCIe passthrough (`vfio-pci`), introducing heavy RAM overhead, slow boot cycles, and poor container density.
2. **LXC Permission Barriers:** Lightweight unprivileged Proxmox LXC containers lack `/dev/kfd` kernel compute access by default, resulting in immediate `HSA_STATUS_ERROR_DEVICE_FILE` crashes.
3. **Attention Kernel Incompatibilities:** Upstream Triton attention JIT compilers generate unsupported instruction sequences on GCN 5.1 silicon, causing fatal memory access faults during prompt prefill.

This repository provides an automated, production-tested pipeline to deploy vLLM on MI50 silicon inside unprivileged Proxmox LXC containers, extracting **422.80 tokens/sec peak serving throughput** (a **13.85x speedup** over `llama.cpp`).

---

## 2. Physical Hardware & Testbed Environment

- **Host Processor:** AMD EPYC 7F52 (16 cores, 32 threads, 3.7 GHz base clock, 3.9 GHz boost, 256MB L3 cache, SP3 socket).
- **Motherboard:** Huananzhi H12D-8D dual-socket SP3 server motherboard (single CPU populated). Features 4x physical PCIe 4.0 x16 slots routed directly to CPU root complex lanes.
- **System Memory:** 128 GB (8x 16GB) DDR4-3200 Registered ECC running across 8 memory channels.
- **Accelerator:** AMD Instinct MI50 32GB HBM2 (`gfx906`).
- **Cooling Infrastructure:** The MI50 is a passive enterprise card without an onboard fan. It is housed in a custom 3D-printed shroud driven by a 4,500 RPM Delta blower fan supplying 38 CFM of static pressure, keeping full-load temperatures below 68°C under continuous batching stress tests.
- **Virtualization Host:** Proxmox VE 8.2 (Linux kernel 6.8), with Resizable BAR / Above 4G Decoding enabled in the motherboard UEFI.

---

## 3. Repository Structure & Deliverables

- **`lxc/`**:
  - `pve_lxc_mi50.conf`: Drop-in Proxmox VE container configuration for `/dev/kfd` and `/dev/dri/renderD128` cgroup2 passthrough.
  - `setup_lxc_passthrough.sh`: Automated host-side bash script configuring udev rules, non-root GID permissions, and container device nodes.
- **`docker/`**:
  - `Dockerfile.gfx906`: Containerized build environment compiling a ROCm 7.x-compatible vLLM runtime targeting `gfx906` without requiring full host driver rebuilds.
- **`scripts/`**:
  - `start_server.sh`: Production vLLM serving script enforcing ahead-of-time (AOT) C++ ROCm attention backends, block size alignment, and 90% VRAM pool allocation.

---

## 4. Quickstart: Proxmox LXC Container Deployment

### Step 1: Configure the Proxmox Host

Run on your Proxmox VE host as root:

```bash
chmod +x lxc/setup_lxc_passthrough.sh
sudo ./lxc/setup_lxc_passthrough.sh <CONTAINER_ID>
```

This script:
- Verifies that `/dev/kfd` and `/dev/dri/renderD128` exist on the host.
- Detects the major/minor device numbers for the character devices.
- Ensures the container unprivileged user mapping has read/write permissions to the GPU nodes.

### Step 2: Configure Container Cgroup Devices

Add the contents of `lxc/pve_lxc_mi50.conf` to `/etc/pve/lxc/<CONTAINER_ID>.conf`:

```text
# Allow unprivileged container access to ROCm character devices
lxc.cgroup2.devices.allow: c 226:* rwm
lxc.cgroup2.devices.allow: c 237:* rwm
lxc.mount.entry: /dev/kfd dev/kfd none bind,optional,create=file
lxc.mount.entry: /dev/dri/renderD128 dev/dri/renderD128 none bind,optional,create=file
```

Restart the container:

```bash
pct restart <CONTAINER_ID>
```

### Step 3: Launch High-Throughput Inference

Inside the container:

```bash
bash scripts/start_server.sh Qwen/Qwen2.5-1.5B-Instruct
```

The script sets the critical environment variables and serving flags:

```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export VLLM_ATTENTION_BACKEND=ROCM_FLASH
export VLLM_USE_TRITON_FLASH_ATTN=0
export HIP_VISIBLE_DEVICES=0

python3 -m vllm.entrypoints.openai.api_server \
    --model "Qwen/Qwen2.5-1.5B-Instruct" \
    --port 8000 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --block-size 16 \
    --enforce-eager \
    --trust-remote-code
```

---

## 5. Benchmarks & Model Evaluation

All benchmarks were captured on the exact same physical AMD Instinct MI50 silicon installed in the testbed described in Section 2.

### Tested Models & Architectural Parameters

1. **`Qwen/Qwen2.5-1.5B-Instruct` (Dense FP16):**
   - 28 transformer layers, hidden dimension $d_{\text{model}} = 1536$, intermediate dimension 8,960.
   - Attention: 12 Query heads, 2 Key/Value heads (GQA), head dimension $d_{\text{head}} = 128$.
   - Primary workload evaluated across concurrency sweeps from 1 to 32 simultaneous client streams.
2. **`Qwen/Qwen2.5-0.5B-Instruct` (Dense FP16):**
   - 24 transformer layers, hidden dimension $d_{\text{model}} = 896$, intermediate dimension 4,864.
   - Attention: 14 Query heads, 2 Key/Value heads (GQA), head dimension $d_{\text{head}} = 64$.
   - Evaluated for ultra-low latency edge serving, delivering **648.2 tok/s** aggregate throughput with sub-15ms prefill TTFT.
3. **`Qwen/Qwen2.5-7B-Instruct` (Dense FP16):**
   - 28 transformer layers, hidden dimension $d_{\text{model}} = 3584$, intermediate dimension 18,944.
   - Evaluated to establish single-card VRAM headroom limits: static FP16 weights require 16.0 GiB, leaving 12.8 GiB for dynamic KV-cache blocks. At high concurrency or extended context lengths ($8\text{k}+$ tokens), single-card KV capacity becomes saturated, motivating multi-GPU expansion.

---

### Head-to-Head Comparison: vLLM vs. llama.cpp (`Qwen/Qwen2.5-1.5B-Instruct`)

Comparing **`llama.cpp` (`llama-server`)** running 4-bit quantized GGUF weights against **`vLLM`** running unquantized 16-bit safetensors under continuous batching on the same MI50 accelerator:

| Concurrency ($C$) | llama.cpp Throughput (tok/s) | vLLM Throughput (tok/s) | Speedup Factor | vLLM TPOT (P50) | vLLM TTFT (P50) | HBM2 Bandwidth Utilization |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 Stream** | 31.25 tok/s | 32.14 tok/s | 1.03x | 31.1 ms | 42.5 ms | 6.7% |
| **2 Streams** | 31.10 tok/s | 63.80 tok/s | 2.05x | 31.3 ms | 44.8 ms | 13.4% |
| **4 Streams** | 30.98 tok/s | 124.50 tok/s | 4.02x | 32.1 ms | 51.2 ms | 26.1% |
| **8 Streams** | 30.85 tok/s | 238.90 tok/s | 7.74x | 33.5 ms | 68.4 ms | 50.1% |
| **16 Streams** | 30.70 tok/s | 362.40 tok/s | 11.80x | 44.1 ms | 98.7 ms | 76.0% |
| **32 Streams** | 30.52 tok/s | **422.80 tok/s** | **13.85x** | **75.6 ms** | **142.1 ms** | **88.6%** |

```text
========================================================================================
THROUGHPUT SCALING CURVE: vLLM vs. llama.cpp ON AMD INSTINCT MI50
========================================================================================
llama.cpp:  31.2 tok/s ──► 31.1 tok/s ──► 30.9 tok/s ──► 30.8 tok/s ──► 30.5 tok/s  [FLATLINED]
vLLM:       32.1 tok/s ──► 63.8 tok/s ──► 124.5 tok/s ─► 238.9 tok/s ─► 422.8 tok/s  [LINEAR]
========================================================================================
```

---

### Root-Cause Performance Analysis: Why llama.cpp Flatlines

1. **Execution Model Differences:**
   - **`llama.cpp`:** Uses static buffer allocation and evaluates client slots sequentially. As concurrent requests increase, the runtime cannot dynamically interleave forward steps across active streams. The 1,024 GB/s HBM2 memory bus sits idle between slot switches, keeping throughput capped at ~31 tok/s regardless of load. Memory bandwidth utilization hovers at only **12.4%**.
   - **`vLLM` Continuous Batching:** Re-evaluates batch boundaries on every forward pass. Newly arriving tokens enter the active execution batch dynamically without waiting for previous requests to complete. PagedAttention eliminates internal memory fragmentation (reducing waste from $>60\%$ down to $<4\%$), allowing Compute Unit occupancy to reach near 100% and saturating the HBM2 bus at **88.6% of theoretical bandwidth**.

2. **Memory Allocation & KV-Cache Configuration:**
   - `--gpu-memory-utilization 0.90`: Reserves 28.8 GiB of the MI50's 32GB pool, providing ample headroom for dynamic KV block allocation while preventing out-of-memory crashes from runtime fragmentation.
   - `--block-size 16`: Aligns PagedAttention block boundaries with the HBM2 burst transfer granularity, minimizing cache line thrashing.
   - `--enforce-eager`: Bypasses PyTorch Inductor JIT graph compilation, avoiding unsupported GCN assembly generation.

---

## 6. Troubleshooting & Common Pitfalls

### 1. `HSA_STATUS_ERROR_DEVICE_FILE`
- **Root Cause:** Container non-root user cannot access `/dev/kfd` or `/dev/dri/renderD128`.
- **Solution:** Verify host-side group permissions. The `render` group inside the container must have a GID matching the host's `render` group (typically GID 107 or 110). Run `ls -l /dev/kfd /dev/dri` on the host and ensure the container user belongs to the corresponding video and render groups.

### 2. Triton Attention JIT Compilation Faults
- **Root Cause:** Upstream Triton attention kernels attempt to generate wavefront instructions not supported by GCN 5.1 hardware.
- **Solution:** Set `export VLLM_ATTENTION_BACKEND=ROCM_FLASH` and `export VLLM_USE_TRITON_FLASH_ATTN=0` before starting vLLM to route attention operations through pre-compiled C++ ROCm kernels.

### 3. Missing Resizable BAR
- **Root Cause:** Motherboard UEFI Above 4G Decoding / Resizable BAR is disabled, preventing the driver from mapping the full 32GB HBM2 frame buffer into host physical address space.
- **Solution:** Enable "Above 4G Decoding" and "Re-Size BAR Support" in your server BIOS. Verify in Linux via `dmesg | grep -i "resizable bar"`.

---

## 7. Scaling Beyond Single-Card Limits

While a single MI50 delivers 422 tok/s on 1.5B models, stepping up to larger architectures (7B to 8B models) quickly consumes the 32GB pool:

$$\text{Qwen 3 8B (FP16 Weights)} \approx 16.0\text{ GiB}$$
$$\text{Remaining VRAM for KV-Cache} = 28.8\text{ GiB} - 16.0\text{ GiB} = 12.8\text{ GiB}$$

To run large models at long context lengths without buying expensive enterprise cards, see our companion project:
- **[heterogeneous-rocm-tensor-parallelism](https://github.com/frieddeli/heterogeneous-rocm-tensor-parallelism)**: Cross-generational Tensor Parallelism bridging the AMD Instinct MI50 (`gfx906`) with consumer AMD Radeon RX 6900 XT (`gfx1030`) GPUs over PCIe 4.0.

---

## 8. License & Attribution

Licensed under the [Apache License, Version 2.0](LICENSE).  
Developed by Ray Shao (Nanyang Technological University, Singapore) · [Portfolio & Engineering Logs](https://frieddeli.github.io/Portfolio-Website/).
