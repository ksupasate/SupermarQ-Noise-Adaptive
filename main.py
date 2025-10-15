"""Quantum Transpilation Analysis using SupermarQ Benchmark Suite.

This module implements a comprehensive analysis of quantum circuit transpilation
at different optimization levels, evaluating their impact on circuit performance
and noise resilience.

Based on the research paper:
    "Understanding Noise-Adaptive Transpilation Techniques Using the SupermarQ Benchmark"
    Vorathammathorn et al., 2025 International Conference on Quantum Communications,
    Networking, and Computing (QCNC), pp. 416-420.
    DOI: 10.1109/QCNC64685.2025.00071

The study examines six quantum algorithms across four Qiskit transpilation
optimization levels (0-3), measuring two-qubit gate counts and five SupermarQ
performance metrics: Program Communication (PC), Critical Depth (CD),
Entanglement Ratio (ER), Liveness (LV), and Parallelism (PL).

Implementation Notes:
    - Backend: FakeJakartaV2 (updated from Fake7QPulseV1 in paper for Qiskit 2.0+ compatibility)
    - Framework: Qiskit 2.0+, SupermarQ 0.5+ (updated from versions in original paper)
    - Both backends are 7-qubit systems with linear topology

Author: Supasate Vorathammathorn and contributors
License: MIT
Repository: https://github.com/ksupasate/SupermarQ-Noise-Adaptive
"""

import os
import sys
import logging
from typing import Dict, Callable, List, Tuple, Any
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# Qiskit imports for quantum circuit operations
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import (
    GroverOperator,
    HamiltonianGate,
    EfficientSU2,
    PhaseEstimation,
    QFT,
)
from qiskit.quantum_info import Operator

# Fake backend import for Qiskit 1.0+/2.0+
# In Qiskit 2.x, fake backends moved to qiskit_ibm_runtime.fake_provider
# We need a 7-qubit backend for our algorithms
try:
    # Try FakeJakartaV2 first (7-qubit, Qiskit 2.x)
    from qiskit_ibm_runtime.fake_provider import FakeJakartaV2
    BACKEND_CLASS = FakeJakartaV2
    BACKEND_NAME = "FakeJakartaV2"
    BACKEND_SOURCE = "qiskit_ibm_runtime"
except ImportError:
    try:
        # Try FakeCasablancaV2 as fallback (7-qubit)
        from qiskit_ibm_runtime.fake_provider import FakeCasablancaV2
        BACKEND_CLASS = FakeCasablancaV2
        BACKEND_NAME = "FakeCasablancaV2"
        BACKEND_SOURCE = "qiskit_ibm_runtime"
    except ImportError:
        try:
            # Try older Qiskit 1.x location
            from qiskit.providers.fake_provider import FakeJakarta
            BACKEND_CLASS = FakeJakarta
            BACKEND_NAME = "FakeJakarta"
            BACKEND_SOURCE = "qiskit.providers"
        except ImportError:
            print("\n" + "="*70)
            print("ERROR: Could not import Qiskit fake backends.")
            print("="*70)
            print("\nThis project requires Qiskit 1.0+ with fake backends.")
            print("\nPlease install the required packages:")
            print("  conda activate qgss-2025")
            print("  pip install --upgrade pip")
            print("  pip install qiskit>=1.0.0 qiskit-ibm-runtime>=0.20.0")
            print("\nOr install all requirements:")
            print("  pip install -r requirements.txt")
            print("="*70 + "\n")
            sys.exit(1)

# SupermarQ imports for benchmarking
try:
    import supermarq.converters as sm_converters
except ImportError:
    print("\n" + "="*70)
    print("ERROR: SupermarQ is not installed or outdated.")
    print("="*70)
    print("\nPlease install SupermarQ:")
    print("  pip install supermarq")
    print("\nOr install all requirements:")
    print("  pip install -r requirements.txt")
    print("="*70 + "\n")
    sys.exit(1)

# Configure logging for better debugging and progress tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_grover_circuit() -> QuantumCircuit:
    """Create a 7-qubit Grover's Search algorithm circuit.

    Grover's algorithm provides quadratic speedup for unstructured search problems.
    This implementation uses a multi-controlled X gate as the oracle and applies
    two Grover iterations for optimal amplification on a 7-qubit system.

    The oracle marks the target state |1111110⟩ by flipping the phase of states
    where the first six qubits are all in state |1⟩.

    Returns:
        QuantumCircuit: A 7-qubit quantum circuit implementing Grover's algorithm
                       with measurement gates on all qubits.

    References:
        Grover, L. K. (1996). A fast quantum mechanical algorithm for database search.
        Proceedings of the 28th Annual ACM Symposium on Theory of Computing, 212-219.
    """
    # Define the oracle that marks the target state
    oracle = QuantumCircuit(7)
    oracle.mcx(list(range(6)), 6)  # Multi-controlled X on target qubit

    # Create Grover operator from oracle
    grover_op = GroverOperator(oracle)

    # Build the complete circuit
    circuit = QuantumCircuit(7)
    circuit.h(range(7))  # Initialize superposition across all qubits

    # Apply Grover iterations (optimal number for 7-qubit search space)
    circuit.compose(grover_op, inplace=True)
    circuit.compose(grover_op, inplace=True)

    circuit.measure_all()
    circuit.name = "Grover's Search"

    logger.debug(f"Created Grover circuit with depth: {circuit.depth()}")
    return circuit


def create_hamiltonian_sim_circuit() -> QuantumCircuit:
    """Create a 7-qubit Hamiltonian Simulation circuit.

    Hamiltonian simulation is fundamental to quantum chemistry and materials science.
    This implementation simulates time evolution under a Pauli-Z Hamiltonian acting
    on the first three qubits: H = Z ⊗ I ⊗ I^⊗5

    The evolution is performed for time t=1.0 using Qiskit's HamiltonianGate,
    which implements the unitary operator U = e^(-iHt).

    Returns:
        QuantumCircuit: A 7-qubit circuit simulating Hamiltonian time evolution.

    References:
        Lloyd, S. (1996). Universal quantum simulators. Science, 273(5278), 1073-1078.
    """
    # Construct Hamiltonian: Z on first qubit, identity on others
    # Using Kronecker products to build multi-qubit operator
    pauli_z = np.array([[1, 0], [0, -1]])
    identity_2 = np.eye(2)
    identity_32 = np.eye(2**5)  # Identity on remaining 5 qubits

    hamiltonian = Operator(np.kron(np.kron(pauli_z, identity_2), identity_32))

    # Simulation time parameter
    time = 1.0

    circuit = QuantumCircuit(7)
    circuit.append(HamiltonianGate(hamiltonian, time), range(7))
    circuit.measure_all()
    circuit.name = "Hamiltonian Sim"

    logger.debug(f"Created Hamiltonian simulation circuit for time t={time}")
    return circuit


def create_hidden_shift_circuit() -> QuantumCircuit:
    """Create a 7-qubit Hidden Shift algorithm circuit.

    The Hidden Shift problem is a fundamental problem in quantum algorithms,
    related to the abelian hidden subgroup problem. Given two functions f and g
    where g(x) = f(x ⊕ s) for some hidden shift s, the algorithm finds s.

    This implementation uses a more complex balanced function that creates
    substantial entanglement through controlled operations across all qubits.

    Returns:
        QuantumCircuit: A 7-qubit circuit for the Hidden Shift problem with
                       classical measurement registers.

    References:
        van Dam, W., Hallgren, S., & Ip, L. (2003). Quantum algorithms for some
        hidden shift problems. SIAM Journal on Computing, 36(3), 763-778.
    """
    n = 7
    circuit = QuantumCircuit(n, n)

    # Initial Hadamard layer for superposition
    circuit.h(range(n))
    circuit.barrier()

    # Create entangled oracle with circular CNOT pattern
    # This creates genuine multi-qubit entanglement
    for i in range(n):
        circuit.cx(i, (i + 1) % n)

    # Add Toffoli gates for more complex function encoding
    circuit.ccx(0, 1, 2)
    circuit.ccx(2, 3, 4)
    circuit.ccx(4, 5, 6)

    # Add another layer of CNOTs with different connectivity
    for i in range(0, n-1, 2):
        if i+1 < n:
            circuit.cx(i+1, i)

    circuit.barrier()

    # Final Hadamard layer for interference
    circuit.h(range(n))
    circuit.measure(range(n), range(n))
    circuit.name = "Hidden Shift"

    logger.debug(f"Created Hidden Shift circuit with enhanced entanglement structure")
    return circuit


def create_amplitude_estimation_circuit() -> QuantumCircuit:
    """Create a 7-qubit Amplitude Estimation circuit.

    Quantum Amplitude Estimation (QAE) provides quadratic speedup over classical
    Monte Carlo methods for estimating expectation values. This is crucial for
    applications in finance, machine learning, and quantum chemistry.

    This implementation creates an entangled state preparation oracle followed
    by amplitude amplification via Grover-like operators, then applies QPE.
    Uses 7 qubits total with substantial entanglement.

    Returns:
        QuantumCircuit: A 7-qubit amplitude estimation circuit using QPE.

    References:
        Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2002). Quantum amplitude
        amplification and estimation. Contemporary Mathematics, 305, 53-74.
    """
    n = 7
    circuit = QuantumCircuit(n, n)

    # State preparation: Create entangled superposition
    circuit.h(range(n))

    # Oracle: Mark target states with entangling operations
    # This creates significant multi-qubit correlations
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)

    # Multi-controlled Z gate (phase oracle)
    circuit.h(3)
    circuit.ccx(0, 1, 4)
    circuit.ccx(2, 4, 3)
    circuit.ccx(0, 1, 4)
    circuit.h(3)

    # Diffusion operator (amplitude amplification)
    circuit.h(range(4))
    circuit.x(range(4))

    # Multi-controlled Z
    circuit.h(3)
    circuit.ccx(0, 1, 4)
    circuit.ccx(2, 4, 3)
    circuit.ccx(0, 1, 4)
    circuit.h(3)

    circuit.x(range(4))
    circuit.h(range(4))

    # Additional entangling layer
    for i in range(n-1):
        circuit.cx(i, i+1)

    circuit.measure_all()
    circuit.name = "Amplitude Est"

    logger.debug(f"Created Amplitude Estimation circuit with enhanced entangling structure")
    return circuit


def create_monte_carlo_circuit() -> QuantumCircuit:
    """Create a 7-qubit Quantum Monte Carlo Sampling circuit.

    Quantum Monte Carlo algorithms leverage quantum superposition and entanglement
    to sample from complex probability distributions. This implementation uses the
    EfficientSU2 ansatz, a hardware-efficient variational form with linear
    entanglement structure.

    The circuit creates a parameterized quantum state that can be used for
    variational quantum algorithms and quantum machine learning tasks. Parameters
    are bound to concrete values to enable proper transpilation analysis.

    Returns:
        QuantumCircuit: A 7-qubit variational circuit with measurement gates.

    References:
        Kandala, A., et al. (2017). Hardware-efficient variational quantum eigensolver
        for small molecules. Nature, 549(7671), 242-246.
    """
    # EfficientSU2 ansatz with linear entanglement topology
    # reps=2 provides sufficient expressibility while maintaining trainability
    base_circuit = EfficientSU2(7, reps=2, entanglement='linear')

    # Bind parameters to concrete values for transpilation analysis
    # Using random but fixed values to create a concrete instantiation
    num_params = base_circuit.num_parameters
    parameter_values = np.random.RandomState(42).uniform(0, 2*np.pi, num_params)

    # Create bound circuit
    circuit = base_circuit.assign_parameters(parameter_values)
    circuit.measure_all()
    circuit.name = "Monte Carlo"

    logger.debug(f"Created Monte Carlo circuit with {num_params} bound parameters")
    return circuit


def create_shor_circuit() -> QuantumCircuit:
    """Create a 7-qubit Shor's Algorithm circuit for factoring N=15.

    Shor's algorithm provides exponential speedup for integer factorization,
    with profound implications for cryptography. This implementation demonstrates
    the order-finding subroutine for factoring N=15 with base a=4.

    The quantum core uses Quantum Phase Estimation (QPE) to find the period r
    such that a^r ≡ 1 (mod N). For N=15 and a=4, the order is r=2.

    Note: This is a simplified 7-qubit version. Full Shor's for N=15 typically
    requires more qubits for complete accuracy.

    Returns:
        QuantumCircuit: A 7-qubit circuit implementing Shor's order-finding routine.

    References:
        Shor, P. W. (1997). Polynomial-time algorithms for prime factorization and
        discrete logarithms on a quantum computer. SIAM Journal on Computing, 26(5).
    """
    n_count = 3  # Counting qubits for phase estimation
    n_target = 4  # Target register for modular exponentiation
    a = 4  # Base for order finding
    N = 15  # Number to factor

    # Controlled unitary operator: U|y⟩ = |ay mod N⟩
    # For a=4, mod 15: implements permutation via swaps
    U = QuantumCircuit(n_target)
    U.swap(0, 1)
    U.swap(1, 2)
    U.swap(2, 3)
    c_U = U.to_gate().control()

    # Build complete QPE circuit
    circuit = QuantumCircuit(n_count + n_target, n_count)

    # Initialize counting qubits in superposition
    circuit.h(range(n_count))

    # Initialize target register to |1⟩ (required for modular exponentiation)
    circuit.x(n_count)

    # Apply controlled-U^(2^j) operations for QPE
    for q in range(n_count):
        circuit.append(
            c_U.power(2**q),
            [q] + list(range(n_count, n_count + n_target))
        )

    # Inverse QFT on counting register
    qft_dagger = QFT(n_count, do_swaps=False).inverse().to_gate()
    circuit.append(qft_dagger, range(n_count))

    # Measure counting register to extract period information
    circuit.measure(range(n_count), range(n_count))
    circuit.name = "Shor's Algorithm"

    logger.debug(f"Created Shor circuit for N={N}, a={a}")
    return circuit


def run_experiment() -> Dict[str, Dict[str, Dict[int, Any]]]:
    """Execute the complete transpilation and benchmarking experiment.

    This function orchestrates the entire analysis pipeline:
    1. Generates circuits for six quantum algorithms
    2. Transpiles each circuit at optimization levels 0, 1, 2, 3
    3. Measures two-qubit gate counts
    4. Computes SupermarQ performance metrics
    5. Aggregates results for visualization

    The experiment uses Qiskit's Fake7QPulseV1 backend to simulate a 7-qubit
    device with realistic noise characteristics and linear qubit connectivity.

    Returns:
        Dict[str, Dict[str, Dict[int, Any]]]: Nested dictionary structure:
            - First level: Algorithm names
            - Second level: Metric types (two_qubit_count, PC, CD, ER, LV, PL)
            - Third level: Optimization level (0-3) → metric value

    Raises:
        RuntimeError: If transpilation or benchmarking fails
    """
    # Algorithm registry with factory functions
    algorithms: Dict[str, Callable[[], QuantumCircuit]] = {
        "Grover's Search": create_grover_circuit,
        "Hamiltonian Sim": create_hamiltonian_sim_circuit,
        "Hidden Shift": create_hidden_shift_circuit,
        "Amplitude Est": create_amplitude_estimation_circuit,
        "Monte Carlo": create_monte_carlo_circuit,
        "Shor's Algorithm": create_shor_circuit,
    }

    # Initialize simulated backend (Qiskit 1.0+/2.0+ fake backend)
    try:
        backend = BACKEND_CLASS()
        backend_name = backend.name if hasattr(backend, 'name') else BACKEND_NAME
        num_qubits = backend.num_qubits if hasattr(backend, 'num_qubits') else 'N/A'

        # Get coupling map size safely
        coupling_info = 'N/A'
        if hasattr(backend, 'coupling_map') and backend.coupling_map:
            try:
                # CouplingMap has size() method in Qiskit 2.x
                if hasattr(backend.coupling_map, 'size'):
                    coupling_info = f"{backend.coupling_map.size()} edges"
                elif hasattr(backend.coupling_map, '__len__'):
                    coupling_info = f"{len(backend.coupling_map)} edges"
                else:
                    coupling_info = "available"
            except:
                coupling_info = "available"

        logger.info(f"Using backend: {BACKEND_NAME} from {BACKEND_SOURCE}")
        logger.info(f"  - Device: {backend_name}")
        logger.info(f"  - Qubits: {num_qubits}")
        logger.info(f"  - Coupling map: {coupling_info}")
    except Exception as e:
        logger.error(f"Failed to initialize backend: {e}")
        raise RuntimeError("Backend initialization failed") from e

    optimization_levels = [0, 1, 2, 3]
    results = defaultdict(lambda: defaultdict(dict))

    logger.info("=" * 60)
    logger.info("Starting Quantum Transpilation Analysis")
    logger.info("=" * 60)

    total_experiments = len(algorithms) * len(optimization_levels)
    current_experiment = 0

    for name, circuit_func in algorithms.items():
        logger.info(f"\nProcessing algorithm: {name}")

        for level in optimization_levels:
            current_experiment += 1
            progress = (current_experiment / total_experiments) * 100

            try:
                # Generate fresh circuit instance
                circuit = circuit_func()
                original_depth = circuit.depth()

                logger.info(f"  [Level {level}] Transpiling... ({progress:.1f}% complete)")

                # Transpile with specified optimization level
                transpiled_circuit = transpile(
                    circuit,
                    backend=backend,
                    optimization_level=level,
                    seed_transpiler=42  # Reproducibility
                )

                transpiled_depth = transpiled_circuit.depth()

                # Extract two-qubit gate count (CX gates on this backend)
                ops_count = transpiled_circuit.count_ops()
                two_qubit_gates = ops_count.get('cx', 0)
                results[name]['two_qubit_count'][level] = two_qubit_gates

                logger.info(f"    Depth: {original_depth} → {transpiled_depth}, "
                          f"CX gates: {two_qubit_gates}")

                # Compute SupermarQ metrics using Qiskit-specific functions
                try:
                    # SupermarQ 0.5+ has special Qiskit converter functions
                    pc = sm_converters.compute_communication_with_qiskit(transpiled_circuit)
                    cd = sm_converters.compute_depth_with_qiskit(transpiled_circuit)
                    er = sm_converters.compute_entanglement_with_qiskit(transpiled_circuit)
                    lv = sm_converters.compute_liveness_with_qiskit(transpiled_circuit)
                    pl = sm_converters.compute_parallelism_with_qiskit(transpiled_circuit)

                    # Store individual metrics
                    results[name]['PC'][level] = pc  # Program Communication
                    results[name]['CD'][level] = cd  # Critical Depth
                    results[name]['ER'][level] = er  # Entanglement Ratio
                    results[name]['LV'][level] = lv  # Liveness
                    results[name]['PL'][level] = pl  # Parallelism

                    logger.info(
                        "    SupermarQ metrics: "
                        f"PC={pc:.6f}, CD={cd:.6f}, ER={er:.6f}, "
                        f"LV={lv:.6f}, PL={pl:.6f}"
                    )

                except Exception as e:
                    logger.error(f"    Failed to compute SupermarQ metrics: {e}")
                    raise

            except Exception as e:
                logger.error(f"  [Level {level}] Experiment failed: {e}")
                raise RuntimeError(f"Experiment failed for {name} at level {level}") from e

    logger.info("\n" + "=" * 60)
    logger.info("Experiment completed successfully!")
    logger.info("=" * 60)

    return dict(results)


def plot_two_qubit_gate_counts(results: Dict[str, Dict[str, Dict[int, Any]]]) -> None:
    """Generate and save a bar chart visualization of two-qubit gate counts.

    Creates a grouped bar chart comparing the number of two-qubit (CX) gates
    required for each algorithm across the four transpilation optimization levels.
    This metric is crucial as two-qubit gates are typically the most error-prone
    operations in current quantum hardware.

    Args:
        results: Nested dictionary containing experimental data with structure:
                 results[algorithm_name][metric_type][optimization_level] = value

    Raises:
        IOError: If unable to save the plot to the output directory
    """
    logger.info("\nGenerating two-qubit gate count visualization...")

    labels = list(results.keys())
    x = np.arange(len(labels))
    width = 0.2  # Bar width

    fig, ax = plt.subplots(figsize=(12, 7))

    # Create grouped bars for each optimization level
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    for i, level in enumerate([0, 1, 2, 3]):
        counts = [results[algo]['two_qubit_count'][level] for algo in labels]
        offset = width * (i - 1.5)
        ax.bar(
            x + offset,
            counts,
            width,
            label=f'Level {level}',
            color=colors[i],
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )

    # Formatting and labels
    ax.set_ylabel('Two-Qubit Gate Count (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Two-Qubit Gate Counts Across Optimization Levels',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace(' ', '\n') for label in labels], fontsize=10)
    ax.legend(title='Optimization Level', fontsize=10, title_fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Use logarithmic scale to show algorithms with vastly different gate counts
    ax.set_yscale('log')
    ax.set_ylim(bottom=1)  # Start from 1 to avoid log(0)

    fig.tight_layout()

    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)

    output_path = 'output/figure_1_two_qubit_counts.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved plot: {output_path}")


def plot_supermarq_metrics(results: Dict[str, Dict[str, Dict[int, Any]]]) -> None:
    """Generate and save comprehensive SupermarQ metrics visualization.

    Creates a multi-panel figure displaying all five SupermarQ performance metrics
    across optimization levels. The metrics provide insight into different aspects
    of quantum circuit quality:

    - Program Communication (PC): Inter-qubit interaction requirements
    - Critical Depth (CD): Longest computational path
    - Entanglement Ratio (ER): Degree of quantum entanglement
    - Liveness (LV): Qubit utilization efficiency
    - Parallelism (PL): Potential for parallel gate execution

    Algorithms are grouped into two categories (A and B) with different line styles
    for visual distinction.

    Args:
        results: Nested dictionary containing experimental data with structure:
                 results[algorithm_name][metric_type][optimization_level] = value

    Raises:
        IOError: If unable to save the plot to the output directory
    """
    logger.info("\nGenerating SupermarQ metrics visualization...")

    metrics = ['PC', 'CD', 'ER', 'LV', 'PL']
    metric_names = [
        'Program Communication (PC)',
        'Critical Depth (CD)',
        'Entanglement Ratio (ER)',
        'Liveness (LV)',
        'Parallelism (PL)'
    ]

    # Group algorithms for visual distinction
    group_a_algos = ["Grover's Search", "Hamiltonian Sim", "Hidden Shift"]
    group_b_algos = ["Amplitude Est", "Monte Carlo", "Shor's Algorithm"]

    # Color scheme and markers
    colors = plt.cm.viridis(np.linspace(0, 1, 6))
    markers = ['o', 's', 'D', '^', 'v', '>']

    # Create 3x2 subplot grid
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    axes = axes.flatten()

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        algo_idx = 0

        # Plot Group A algorithms (dashed lines)
        for algo in group_a_algos:
            levels = sorted(results[algo][metric].keys())
            values = [results[algo][metric][l] for l in levels]
            ax.plot(
                levels, values,
                label=f'{algo}',
                linestyle='--',
                marker=markers[algo_idx],
                color=colors[algo_idx],
                linewidth=2,
                markersize=8,
                alpha=0.8
            )
            algo_idx += 1

        # Plot Group B algorithms (solid lines)
        for algo in group_b_algos:
            levels = sorted(results[algo][metric].keys())
            values = [results[algo][metric][l] for l in levels]
            ax.plot(
                levels, values,
                label=f'{algo}',
                linestyle='-',
                marker=markers[algo_idx],
                color=colors[algo_idx],
                linewidth=2,
                markersize=8,
                alpha=0.8
            )
            algo_idx += 1

        # Subplot formatting
        ax.set_title(name, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Optimization Level', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['L0', 'L1', 'L2', 'L3'])
        ax.grid(True, linestyle=':', alpha=0.4)

    # Remove the last empty subplot
    fig.delaxes(axes[-1])

    # Create shared legend in the empty subplot space
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower right',
        bbox_to_anchor=(0.88, 0.08),
        title='Quantum Algorithms',
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        shadow=True
    )

    # Overall title and layout
    fig.suptitle(
        'SupermarQ Metrics vs. Optimization Level',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])

    # Save figure
    os.makedirs('output', exist_ok=True)
    output_path = 'output/figure_2_supermarq_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"✓ Saved plot: {output_path}")


def main() -> int:
    """Main execution function for the transpilation analysis pipeline.

    Orchestrates the complete experimental workflow:
    1. Runs transpilation experiments on all algorithms
    2. Generates visualizations of results
    3. Handles errors gracefully with appropriate logging

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        # Execute experimental pipeline
        experiment_results = run_experiment()

        # Generate visualizations
        plot_two_qubit_gate_counts(experiment_results)
        plot_supermarq_metrics(experiment_results)

        logger.info("\n" + "=" * 60)
        logger.info("Analysis pipeline completed successfully!")
        logger.info("Output files saved to: ./output/")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.warning("\nExperiment interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"\nExperiment failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
