# Quantum Transpilation Analysis with SupermarQ

<div align="center">

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-6929C4.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![SupermarQ](https://img.shields.io/badge/SupermarQ-0.5+-orange.svg)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FQCNC64685.2025.00071-blue)](https://doi.org/10.1109/QCNC64685.2025.00071)

**A comprehensive benchmarking framework for analyzing quantum circuit transpilation across multiple optimization levels using the SupermarQ benchmark suite**

[Features](#features) •
[Installation](#installation) •
[Quick Start](#quick-start) •
[Tutorial](#interactive-tutorial) •
[Documentation](#documentation) •
[Research Paper](#research-paper) •
[Contributing](#contributing)

</div>

---

## Overview

This project is an **open-source implementation** based on the research paper:

> **"Understanding Noise-Adaptive Transpilation Techniques Using the SupermarQ Benchmark"**
> Vorathammathorn et al., 2025 International Conference on Quantum Communications, Networking, and Computing (QCNC)
> DOI: [10.1109/QCNC64685.2025.00071](https://doi.org/10.1109/QCNC64685.2025.00071)

This implementation provides a systematic analysis of **quantum circuit transpilation** techniques, evaluating how different Qiskit optimization levels affect circuit performance and noise resilience using the SupermarQ benchmarking suite.

### What It Does

- **Implements** six fundamental quantum algorithms (Grover, Hamiltonian Simulation, Hidden Shift, Amplitude Estimation, Monte Carlo, Shor)
- **Transpiles** circuits at four optimization levels (0, 1, 2, 3) using Qiskit
- **Measures** two-qubit gate counts and five SupermarQ performance metrics
- **Visualizes** comparative results with publication-quality plots
- **Provides** an interactive Jupyter notebook for hands-on learning

### Implementation Notes

This implementation is based on the published research paper but includes the following modifications for compatibility with modern quantum computing frameworks:

#### Backend Changes

- **Paper**: Used `Fake7QPulseV1` (Qiskit 0.x)
- **This Implementation**: Uses `FakeJakartaV2` (Qiskit 2.0+)
  - Both are 7-qubit backends with linear topology
  - FakeJakartaV2 provides updated noise models and gate characteristics
  - Results are comparable but may differ slightly due to backend noise model updates

#### Framework Versions

- **Qiskit**: Upgraded from 0.x to 2.0+ (latest stable release)
- **SupermarQ**: Updated from 0.1.0 to 0.5+ with new API
- **Python**: Compatible with Python 3.9+ (3.10+ recommended)

### Research Context

Transpilation optimization is crucial for NISQ-era quantum computers because:

- Two-qubit gates are the primary source of errors in current quantum hardware
- Circuit depth directly impacts decoherence and gate fidelity
- Different algorithms respond differently to optimization strategies
- Hardware topology constraints (qubit connectivity) necessitate SWAP gate insertion

The paper demonstrates how SupermarQ benchmarking metrics reveal algorithm-specific optimization characteristics and noise-resilience patterns.

---

## Features

### 🎯 **Six Quantum Algorithms**
- **Grover's Search** - Quadratic speedup for unstructured search
- **Hamiltonian Simulation** - Quantum system dynamics
- **Hidden Shift** - Abelian hidden subgroup problem
- **Amplitude Estimation** - Quantum advantage for Monte Carlo
- **Monte Carlo Sampling** - Variational quantum algorithms
- **Shor's Algorithm** - Integer factorization

### 📊 **Comprehensive Metrics**
- **Two-Qubit Gate Count** - Primary error source quantification
- **Program Communication (PC)** - Inter-qubit interaction requirements
- **Critical Depth (CD)** - Longest computational path
- **Entanglement Ratio (ER)** - Quantum correlation measure
- **Liveness (LV)** - Qubit utilization efficiency
- **Parallelism (PL)** - Concurrent execution potential

### 📈 **Publication-Quality Visualizations**
- Grouped bar charts comparing gate counts across algorithms
- Multi-panel line plots for SupermarQ metrics
- High-resolution (300 DPI) output for research papers

### 🎓 **Educational Resources**
- Interactive Jupyter notebook with step-by-step tutorials
- Detailed code documentation with quantum computing references
- Hands-on exercises and extension challenges

---

## Installation

### Prerequisites

- **Python 3.9 or higher** (Python 3.10+ recommended)
- **pip package manager** (latest version)
- **Qiskit 1.0 or higher** (required)
- (Optional) Virtual environment manager

### Quick Install

```bash
# Clone the repository
git clone https://github.com/ksupasate/SupermarQ-Noise-Adaptive.git
cd SupermarQ-Noise-Adaptive

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import qiskit; import supermarq; print(f'✓ Installation successful\n  Qiskit version: {qiskit.__version__}\n  Python: {__import__(\"sys\").version.split()[0]}')"
```

**Expected output:**
```
✓ Installation successful
  Qiskit version: 1.x.x
  Python: 3.10.x (or higher)
```

---

## Quick Start

### Run Complete Analysis

```bash
python main.py
```

This will:
1. Generate circuits for all six algorithms
2. Transpile at optimization levels 0, 1, 2, 3
3. Compute gate counts and SupermarQ metrics
4. Save visualizations to `output/` directory

**Expected output:**
```
output/
├── figure_1_two_qubit_counts.png
└── figure_2_supermarq_metrics.png
```

### Example Output

The analysis produces two publication-quality figures:

**Figure 1: Two-Qubit Gate Counts**
- Compares CX gate requirements across algorithms and optimization levels
- Shows the impact of transpilation on error-prone operations

**Figure 2: SupermarQ Metrics**
- Five-panel visualization of circuit quality metrics
- Reveals algorithm-specific optimization characteristics

---

## Interactive Tutorial

### Launch Jupyter Notebook

```bash
jupyter notebook tutorial.ipynb
```

The tutorial provides:

- **Section 1**: Environment setup and verification
- **Section 2**: Understanding quantum transpilation
- **Section 3**: Deep-dive into quantum algorithms
- **Section 4**: Hands-on transpilation analysis
- **Section 5**: SupermarQ benchmarking walkthrough
- **Section 6**: Complete experiment execution
- **Section 7**: Results interpretation and best practices
- **Section 8**: Extension ideas and challenges

### 🎯 Tutorial Highlights

- **Interactive Code Cells** - Run and modify experiments in real-time
- **Visualizations** - See circuit diagrams and performance plots
- **Exercises** - Practice implementing custom algorithms
- **Theory + Practice** - Quantum computing concepts with hands-on examples

---

## Documentation

### Project Structure

```
SupermarQ-Noise-Adaptive/
├── .gitattributes          # Git configuration for notebooks
├── .gitignore              # Ignore patterns for Python/Jupyter
├── CONTRIBUTING.md         # Contribution guidelines
├── LICENSE                 # MIT License
├── README.md               # This file
├── main.py                 # Main analysis pipeline
├── tutorial.ipynb          # Interactive Jupyter tutorial
├── requirements.txt        # Python dependencies
├── requirements-dev.txt    # Development dependencies
└── output/                 # Generated plots (created at runtime)
    ├── figure_1_two_qubit_counts.png
    └── figure_2_supermarq_metrics.png
```

### Algorithm Descriptions

#### Grover's Search
- **Purpose**: Unstructured search with O(√N) complexity
- **Implementation**: 7-qubit circuit with 2 Grover iterations
- **Key Feature**: Multi-controlled X oracle for state marking

#### Hamiltonian Simulation
- **Purpose**: Simulate quantum system time evolution
- **Implementation**: Pauli-Z Hamiltonian on 7 qubits
- **Key Feature**: Implements U = e^(-iHt) via HamiltonianGate

#### Hidden Shift
- **Purpose**: Find hidden shift s where g(x) = f(x ⊕ s)
- **Implementation**: Hadamard-based algorithm with secret '1010101'
- **Key Feature**: Quantum Fourier analysis of Boolean functions

#### Amplitude Estimation
- **Purpose**: Quantum speedup for amplitude/expectation value estimation
- **Implementation**: Quantum Phase Estimation with 5+2 qubits
- **Key Feature**: Quadratic advantage over classical Monte Carlo

#### Monte Carlo Sampling
- **Purpose**: Variational quantum algorithms and QML
- **Implementation**: EfficientSU2 ansatz with linear entanglement
- **Key Feature**: Hardware-efficient parameterized circuits

#### Shor's Algorithm
- **Purpose**: Integer factorization with exponential speedup
- **Implementation**: Simplified 7-qubit order-finding for N=15, a=4
- **Key Feature**: Quantum Phase Estimation for period finding

### Transpilation Optimization Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| **0** | Minimal optimization - basic gate translation | Debugging, understanding circuit structure |
| **1** | Light optimization - simple commutation rules | Fast transpilation with basic improvements |
| **2** | Medium optimization - more aggressive simplification | Balanced performance vs compilation time |
| **3** | Heavy optimization - extensive circuit rewriting | Maximum optimization for production runs |

### SupermarQ Metrics Explained

- **PC (Program Communication)**: Quantifies the communication requirements between qubits. Higher values indicate more inter-qubit dependencies.

- **CD (Critical Depth)**: Measures the longest path of dependent operations. Directly related to circuit execution time and decoherence susceptibility.

- **ER (Entanglement Ratio)**: Proportion of entangling (two-qubit) gates. Indicates the "quantumness" of the algorithm.

- **LV (Liveness)**: Tracks how efficiently qubits are utilized throughout execution. Higher values mean better resource usage.

- **PL (Parallelism)**: Potential for concurrent gate execution. Higher parallelism can reduce effective circuit depth.

---

## Results & Analysis

### Key Findings

1. **Optimization Level Impact Varies by Algorithm**
   - Grover's algorithm shows 30-40% gate count reduction at Level 3
   - Hamiltonian simulation benefits minimally from optimization
   - Shor's algorithm demonstrates complex optimization trade-offs

2. **Two-Qubit Gates are the Bottleneck**
   - CX gates constitute 50-70% of transpiled circuits
   - Higher optimization levels generally reduce CX count
   - But SWAP insertion for connectivity can increase gates

3. **SupermarQ Metrics Reveal Quality**
   - PC correlates with qubit connectivity requirements
   - ER remains relatively stable across optimization levels
   - PL increases with successful optimization

### Performance Considerations

- **Transpilation Time**: Level 3 can take 10x longer than Level 0
- **Hardware Fidelity**: Fewer gates ≠ always better (gate-specific error rates matter)
- **Algorithm Characteristics**: Some algorithms are naturally optimization-resistant

---

## Troubleshooting

### Qiskit Version Requirements

**This project requires Qiskit 1.0 or higher.**

If you have an older version installed:
```bash
# Check your Qiskit version
python -c "import qiskit; print(qiskit.__version__)"

# Upgrade to Qiskit 1.0+
pip install --upgrade qiskit>=1.0.0 qiskit-ibm-runtime>=0.20.0
```

### ImportError: cannot import Fake Backend

**Issue**: Getting import errors for fake backends.

**Solution**: The project uses `FakeManilaV2` (5-qubit) or `FakeJakarta` (7-qubit) backends from Qiskit 1.0+.

```bash
# Install required packages
pip install --upgrade pip
pip install qiskit>=1.0.0 qiskit-ibm-runtime>=0.20.0

# Or install all requirements
pip install -r requirements.txt
```

### SupermarQ Installation Issues

If SupermarQ installation fails:
```bash
# Try installing from source
git clone https://github.com/Infleqtion/client-superstaq.git
cd client-superstaq/supermarq-benchmarks
pip install -e .
```

### Jupyter Notebook Kernel Issues

```bash
# Register the kernel
python -m ipykernel install --user --name quantum-transpilation

# Launch Jupyter
jupyter notebook tutorial.ipynb
```

---

## Advanced Usage

### Custom Backend

```python
# The code automatically uses compatible backends
# You can also specify a custom backend:

from qiskit.providers.fake_provider import FakeManilaV2

# Use a different backend
backend = FakeManilaV2()

# Run experiment (modify run_experiment() to accept backend parameter)
```

### Analyze Single Algorithm

```python
from qiskit import transpile
from main import create_grover_circuit
import supermarq as sm

# Create circuit
circuit = create_grover_circuit()

# Transpile at specific level
transpiled = transpile(circuit, backend=backend, optimization_level=2)

# Get metrics
features = sm.benchmark.get_application_feature_vector(transpiled)
print(f"PC={features[0]}, CD={features[1]}, ER={features[2]}")
```

### Extend with New Algorithms

```python
def create_custom_algorithm():
    """Add your own quantum algorithm."""
    circuit = QuantumCircuit(7)
    # ... your implementation ...
    circuit.measure_all()
    return circuit

# Add to algorithms dictionary in main.py
algorithms["Custom Algorithm"] = create_custom_algorithm
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute

- 🐛 **Bug Reports** - Found an issue? Let us know!
- ✨ **New Features** - Additional algorithms or metrics
- 📖 **Documentation** - Improve tutorials and examples
- 🧪 **Testing** - Expand test coverage
- 🎨 **Visualizations** - Enhanced plotting capabilities

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code formatter
black main.py

# Run type checker
mypy main.py

# Run linter
flake8 main.py
```

---

## Research Paper

This implementation is based on the following peer-reviewed research:

### Publication Details

- **Title**: Understanding Noise-Adaptive Transpilation Techniques Using the SupermarQ Benchmark
- **Conference**: 2025 International Conference on Quantum Communications, Networking, and Computing (QCNC)
- **Authors**: Supasate Vorathammathorn, Muhummud Binhar, Natchapol Patamawisut, Suthep Chanchuphol, Prapong Prechaprapranwong, Jaturon Hansomboon, Rajchawit Sarochawikasit
- **Pages**: 416-420
- **Year**: 2025
- **DOI**: [10.1109/QCNC64685.2025.00071](https://doi.org/10.1109/QCNC64685.2025.00071)
- **Keywords**: Quantum Computing, Noise-Adaptive Transpilation, SupermarQ Benchmarking Suite, Quantum Performance Metrics, Quantum Algorithms

### Abstract Summary

The paper investigates how different transpilation optimization levels affect quantum circuit performance and noise resilience across six quantum algorithms (Grover's Search, Hamiltonian Simulation, Hidden Shift, Amplitude Estimation, Monte Carlo Sampling, and Shor's Algorithm). Using the SupermarQ benchmarking suite, the study reveals algorithm-specific patterns in how optimization impacts circuit quality metrics including Program Communication, Critical Depth, Entanglement Ratio, Liveness, and Parallelism.

---

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@INPROCEEDINGS{11000169,
  author={Vorathammathorn, Supasate and Binhar, Muhummud and Patamawisut, Natchapol and Chanchuphol, Suthep and Prechaprapranwong, Prapong and Hansomboon, Jaturon and Sarochawikasit, Rajchawit},
  booktitle={2025 International Conference on Quantum Communications, Networking, and Computing (QCNC)},
  title={Understanding Noise-Adaptive Transpilation Techniques Using the SupermarQ Benchmark},
  year={2025},
  volume={},
  number={},
  pages={416-420},
  keywords={Quantum computing;Quantum algorithm;Program processors;Noise;Qubit;Logic gates;Benchmark testing;Parallel processing;Optimization;Resilience;Quantum Computing;Noise-Adaptive Transpilation;SupermarQ Benchmarking Suite;Quantum Performance Metrics;Quantum Algorithms},
  doi={10.1109/QCNC64685.2025.00071}
}
```

You may also cite this software implementation:

```bibtex
@software{quantum_transpilation_impl,
  author = {Vorathammathorn, Supasate and contributors},
  title = {SupermarQ Noise-Adaptive Transpilation: Open-Source Implementation},
  year = {2025},
  url = {https://github.com/ksupasate/SupermarQ-Noise-Adaptive},
  note = {Implementation based on Vorathammathorn et al., QCNC 2025},
  license = {MIT}
}
```

Please also consider citing the SupermarQ benchmark suite:

```bibtex
@article{Tomesh2022,
  title={SupermarQ: A Scalable Quantum Benchmark Suite},
  author={Tomesh, Teague and Pranav Gokhale and Victory Omole and others},
  journal={arXiv preprint arXiv:2202.11045},
  year={2022}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **IBM Qiskit Team** - For the excellent quantum computing framework
- **SupermarQ Team** - For the comprehensive benchmarking suite
- **Quantum Computing Community** - For continuous innovation and open collaboration

---

## Support & Contact

- 📫 **Issues**: [GitHub Issues](https://github.com/ksupasate/SupermarQ-Noise-Adaptive/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/ksupasate/SupermarQ-Noise-Adaptive/discussions)
- 📖 **Documentation**: See [tutorial.ipynb](tutorial.ipynb) for detailed guidance

---

## Roadmap

### Planned Features

- [ ] Support for additional quantum backends (IonQ, Rigetti)
- [ ] Extended algorithm library (VQE, QAOA, QPE variants)
- [ ] Real-time hardware execution with IBM Quantum
- [ ] Advanced error mitigation techniques
- [ ] Automated parameter optimization
- [ ] Web-based visualization dashboard
- [ ] Comprehensive unit test suite
- [ ] Continuous integration (CI/CD) pipeline

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

**🔬 Happy Quantum Computing! 🚀**

Made with ❤️ for the quantum computing community

</div>
