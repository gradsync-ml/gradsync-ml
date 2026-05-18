# ⚡️ GradSync

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)

GradSync is a cross-machine distributed training framework for running PyTorch models across a small cluster of heterogeneous consumer hardware. It allows research teams to pool the VRAM of their collective machines (MacBooks, Windows PCs with RTX GPUs, Linux machines) into a single, unified training cluster.

The core framework handles cluster election, dynamic model layer partitioning, and executing pipeline-parallel training over gRPC.

## ✨ Features

- **Pipeline-Parallel Training:** Train models larger than a single machine's VRAM across multiple nodes.
- **Zero-Config Topology:** Automatic cluster election and ordered topology creation via a custom consensus algorithm.
- **Dynamic Sharding:** Model-layer partitioning based on node capacity.
- **Cross-OS Compatibility:** Seamlessly train across Metal (MPS), CUDA, and CPU devices in the same pipeline.
- **Network Optimized:** gRPC transport for forward activations and backward gradients.

## 🛠 Prerequisites

- Python 3.9 or newer.
- [uv](https://github.com/astral-sh/uv) for lightning-fast dependency and environment management.
- Network connectivity between every machine in the training cluster.
- Open firewall access for the election port, training port, and telemetry ports used by the head node.

## 🚀 Setup & Installation

Clone the repository and set up the `uv` virtual environment. GradSync is installed as an editable package so it can be imported globally from anywhere in the project.

```bash
git clone [https://github.com/YOUR-USERNAME/gradsync-ml.git](https://github.com/YOUR-USERNAME/gradsync-ml.git)
cd gradsync-ml

# Initialize the virtual environment and install dependencies
uv venv
uv pip install -e ".[dev]"
