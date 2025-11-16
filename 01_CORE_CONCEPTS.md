# System Blueprint: 01 - Core Concepts

This document defines the key abstract nouns of the DynamicLoRA system.

## Fundamental Abstractions

### LoRA Layer
A **LoRA Layer** is a low-rank adapter that modifies a subset of the base model's weights. In standard LoRA:
```
W_effective = W_base + ΔW
where ΔW = A · B (A, B are learned low-rank matrices)
```

In DynamicLoRA, each layer additionally maintains:
- **Base Inertia** (α_base): Initial resistance to change (configuration parameter)
- **Effective Inertia** (α_eff): Computed from interaction density (emergent property)
- **Screening Threshold** (τ_screen): Activity level triggering phase transition to SCREENED state
- **Interaction History**: Rolling window of activation magnitudes and frequencies
- **Entropy Budget**: Min/max bounds on parameter dispersion rate

### Layer Hierarchy
The **Layer Hierarchy** is the vertical stack of LoRA layers, ordered by temporal scale:

```
Foundation (slowest)
    ↓
Domain
    ↓
Swarm
    ↓
Agent
    ↓
Context (fastest)
```

**Key property:** Each layer is **sorazmerny** (commensurate) with the base model—same dimensionality, but different information density. A programming expert has high density in `LoRA_domain[programming]` but near-zero in `LoRA_domain[medicine]`.

### Phase State
A parameter's **Phase State** determines its behavior:

**SCREENED:**
- High activity density (above screening threshold)
- Forms attractor in parameter space (locked by interaction loops)
- Resists drift (very low decay rate)
- Example: Frequently used circuit for parsing JSON

**FREE:**
- Low activity density (below screening threshold)
- No attractor, free dispersion
- Drifts toward base state (high decay rate)
- Example: Rarely activated parameter, gradually forgotten

**Phase Transition:**
The transition is **sharp** (not smooth). This is a critical property preventing the system from getting stuck in intermediate states. There's a critical threshold; crossing it flips the mode.

Optional: **Hysteresis** (different thresholds for entering vs. exiting SCREENED state) prevents rapid oscillation.

### Circuit
A **Circuit** is a connected subgraph of neurons that performs a specific function. Unlike individual weights, circuits are:
- **Interpretable:** "This circuit handles pronoun resolution"
- **Compositional:** Circuits combine to form higher-level functions
- **Migratable:** Circuits, not weights, move between layers

**Structure:**
```python
Circuit:
    nodes: Set[Neuron]           # Active neurons in the pattern
    edges: Set[(Neuron, Neuron)] # Strong connections (high weight magnitude)
    activation_pattern: Tensor   # Expected activation values
    function: str                # Inferred purpose (optional, for interpretability)
```

**Detection:** Circuits are identified through trajectory analysis (see WaveGen integration). High coherence + repeated activation pattern = candidate circuit.

### Autocorrelation Signal
The **Autocorrelation Signal** measures whether a pattern repeats over time:

```
autocorr(layer, τ) = correlation(
    pattern_signature(t),
    pattern_signature(t - τ)
)
```

Where `pattern_signature` is a hash or embedding of dominant gradient directions.

**Threshold:** If `autocorr > τ_stable` for N consecutive windows, the pattern is considered **stable** and becomes a candidate for upward migration.

**Critical difference from simple frequency:**
- Frequency: "This circuit activated 100 times"
- Autocorrelation: "This circuit activated in the same way across distant time points"

The latter is a much stronger signal of genuine stability.

### Vertical Resonance
**Vertical Resonance** occurs when gradients across multiple layers align (point in the same direction in parameter space):

```python
vertical_alignment = mean([
    dot(grad_context, grad_agent),
    dot(grad_agent, grad_swarm),
    dot(grad_context, grad_swarm)
])
```

**High vertical alignment** (> threshold) indicates a **universal pattern**—useful at all temporal scales. These patterns get fast-tracked for migration to slower layers.

**Low/negative alignment** indicates conflict between layers. In this case:
- Fast layers are dampened (trust the slow layers)
- Pattern does not migrate upward
- Conflict is resolved locally

### Horizontal Resonance
**Horizontal Resonance** occurs when multiple agents' gradients align on a shared layer (Swarm, Domain):

```python
for each new gradient_i:
    alignment = dot(gradient_i, current_swarm_direction)

    if alignment > 0:
        resonance_factor = 1 + α · alignment  # amplify
    else:
        resonance_factor = 1 + β · alignment  # dampen (β < 0)

    weighted_gradient = gradient_i × resonance_factor
```

**Effect:**
- Aligned contributions amplify (constructive interference)
- Opposing contributions cancel (destructive interference)
- Consensus emerges without voting

### Interaction Density
**Interaction Density** is the key emergent property that determines effective inertia:

```python
interaction_density(layer, window) = (
    agent_count(layer) ×
    activation_frequency(layer) ×
    gradient_magnitude(layer)
)

effective_inertia(layer) = base_inertia + (1 - base_inertia) × tanh(interaction_density / k)
```

**Result:** A layer used by 100 agents becomes "heavier" (higher inertia, slower to change) than the same layer used by 5 agents. Temporal scale is not configured—it **emerges** from usage.

### Migration Event
A **Migration Event** is the process of moving a circuit from one layer to the next slower layer:

```
Circuit detected in Layer_fast (via autocorrelation)
    ↓
Check vertical resonance (is it universal?)
    ↓
Check bandwidth availability (can we migrate now?)
    ↓
If yes:
    - Strengthen circuit in Layer_slow
    - Weaken circuit in Layer_fast (gradual handoff)
    - Record migration (for interpretability)
```

**Not instantaneous:** Migration respects bandwidth limits (see below).

### Bandwidth Budget
Each **migration channel** (Context→Agent, Agent→Swarm, etc.) has a bandwidth limit:

```python
bandwidth[layer_i → layer_j] = fraction of parameters per step

Example:
    Context → Agent:  0.01  (1% per step)
    Agent → Swarm:    0.005 (0.5% per step)
    Swarm → Domain:   0.001 (0.1% per step)
```

**Purpose:** Prevent migration avalanche (too many patterns migrating at once, destabilizing target layer).

**Mechanism:** Migration queue (priority = autocorrelation strength).

**Exception:** Emergency bypass for critically important patterns (criticality > threshold).

### Entropy Budget
The **Entropy Budget** maintains homeostasis:

```python
current_entropy(layer) = sum(
    param.dispersion_rate × param.variance
    for param in layer.parameters()
)

If current_entropy > max_entropy:
    # Too fluid (amnesia risk)
    layer.screening_threshold *= 0.9  # easier to enter SCREENED

If current_entropy < min_entropy:
    # Too rigid (crystallization risk)
    layer.screening_threshold *= 1.1  # harder to enter SCREENED
```

**Emergent thresholds:** `min_entropy` and `max_entropy` can themselves adapt based on task complexity (not forced in v1.0, but planned for v2.0).

### Snapshot
A **Snapshot** is a frozen copy of the LoRA hierarchy at a specific point in time:

```python
snapshot = {
    'foundation': lora_foundation.clone(),
    'domain': lora_domain.clone(),
    'swarm': lora_swarm.clone(),
    'agent': lora_agent.clone(),
    'context': lora_context.clone(),
    'timestamp': current_time,
    'provenance_hash': hash(model_id, config, lora_states)
}
```

**Use case:** Inference runs on a snapshot (frozen state), while updates happen asynchronously. This ensures:
- Deterministic inference (same input → same output during single request)
- No race conditions (LoRA can't change mid-forward-pass)
- Easier WaveGen integration (cache keys include snapshot hash)

**Trade-off:** Updates apply between requests, not during. Live updates are possible but more complex (see 02_ARCHITECTURE.md for discussion).

## Derived Concepts

### Universality Score
Combines vertical and horizontal resonance to detect fundamental patterns:

```python
universality_score = vertical_alignment × horizontal_consensus

If universality_score > threshold:
    # This pattern is useful everywhere
    fast_track_migration(pattern, target=domain_or_foundation)
```

### Consolidation Pressure
How urgently a pattern "wants" to migrate:

```python
consolidation_pressure(circuit) = (
    autocorr(circuit) ×          # stability over time
    usage_frequency(circuit) ×   # how often used
    vertical_resonance(circuit)  # universality
)
```

Patterns with high pressure get priority in migration queue.

### Layer Health Metrics
Per-layer indicators:

```python
health(layer) = {
    'entropy_ratio': current_entropy / optimal_entropy(layer),
    'screening_rate': fraction_of_params_in_SCREENED_state,
    'migration_backlog': len(migration_queue[layer]),
    'conflict_rate': rejected_updates / total_updates,
    'effective_inertia': computed_inertia(layer)
}
```

Healthy system:
- `0.8 < entropy_ratio < 1.2`
- `0.4 < screening_rate < 0.7` (not too rigid, not too fluid)
- `migration_backlog < 10`
- `conflict_rate < 0.05`

## WaveGen Integration Points

### Cache Hit Rate → LoRA Update Signal
**WaveGen** tracks trajectory cache statistics. High hit rate = frequently used circuit.

```python
if wavegen_cache[circuit].hit_rate > migration_threshold:
    # This circuit is stable enough to internalize
    lora_layer.consolidate(circuit)
    wavegen_cache.evict(circuit)  # free cache space
```

### Circuit Extraction
**WaveGen** detects stable trajectories through coherence coefficient. **DynamicLoRA** extracts the underlying circuit (activation pattern) from those trajectories.

```python
trajectory = wavegen.observe_generation(prompt)
if trajectory.coherence > threshold:
    circuit = extract_circuit(trajectory.activation_history)
    lora_context.consider_for_migration(circuit)
```

### Provenance Alignment
Both systems use provenance hashes to ensure cache/LoRA consistency:

```python
wavegen_cache_key = hash(prompt + lora_snapshot_hash)
lora_provenance = hash(model_id + config + layer_states)

# These must match for cache lookup to succeed
```

---

**Key Insight:** DynamicLoRA doesn't work *on* individual weights—it works on **circuits** (interpretable activation patterns). WaveGen detects these circuits through trajectory observation. Together, they form a complete learning system: observation (WaveGen) + consolidation (LoRA).
