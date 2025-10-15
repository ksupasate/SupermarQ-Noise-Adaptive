# Examples Directory

This directory contains example outputs and supplementary materials for the Quantum Transpilation Analysis project.

## Contents

### Generated Plots

When you run `python main.py`, the following plots will be generated in the `../output/` directory:

- **figure_1_two_qubit_counts.png** - Bar chart comparing two-qubit (CX) gate counts across six algorithms and four optimization levels
- **figure_2_supermarq_metrics.png** - Multi-panel line plots showing five SupermarQ metrics (PC, CD, ER, LV, PL) across optimization levels

### Example Visualizations

This directory can be used to store:
- Sample circuit diagrams
- Comparative analysis plots
- Custom visualization examples
- Extended benchmarking results

## Usage Examples

### Visualize a Single Circuit

```python
from qiskit.visualization import circuit_drawer
from main import create_grover_circuit

# Create and visualize Grover circuit
circuit = create_grover_circuit()
circuit.draw('mpl', filename='examples/grover_circuit.png')
```

### Compare Transpiled Circuits

```python
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import Fake7QPulseV1
from main import create_grover_circuit
import matplotlib.pyplot as plt

backend = Fake7QPulseV1()
original = create_grover_circuit()

# Transpile at different levels
level_0 = transpile(original, backend=backend, optimization_level=0)
level_3 = transpile(original, backend=backend, optimization_level=3)

# Create comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

original.draw('mpl', ax=ax1)
ax1.set_title('Level 0')

level_3.draw('mpl', ax=ax2)
ax2.set_title('Level 3')

plt.tight_layout()
plt.savefig('examples/transpilation_comparison.png', dpi=300)
```

### Export Circuit to QASM

```python
from main import create_grover_circuit

circuit = create_grover_circuit()
qasm_str = circuit.qasm()

with open('examples/grover_circuit.qasm', 'w') as f:
    f.write(qasm_str)
```

## Adding Your Examples

Feel free to add your own example notebooks, scripts, or visualizations to this directory. When contributing examples:

1. Include clear documentation
2. Provide context for what the example demonstrates
3. Use descriptive filenames
4. Ensure examples are self-contained and reproducible

## Resources

- Main analysis: `../main.py`
- Interactive tutorial: `../tutorial.ipynb`
- Documentation: `../README.md`
