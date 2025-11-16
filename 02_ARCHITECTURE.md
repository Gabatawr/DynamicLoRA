# System Blueprint: 02 - Component Architecture

This document details the individual components of the DynamicLoRA system and their responsibilities.

## Core Components

### 1. `LoRAHierarchy` (Central Coordinator)

The main orchestrator managing the vertical stack of LoRA layers.

**Responsibilities:**
- Initialize and maintain all LoRA layers (Foundation, Domain, Swarm, Agent, Context)
- Create snapshots for inference
- Route migration requests between layers
- Enforce bandwidth limits
- Monitor system health (entropy budget, conflict rates)

**Key Methods:**
```python
class LoRAHierarchy:
    def __init__(self, base_model, config):
        self.base = base_model  # frozen
        self.layers = {
            'foundation': DynamicLoRALayer(rank=64, base_inertia=0.99),
            'domain': DynamicLoRALayer(rank=32, base_inertia=0.9),
            'swarm': DynamicLoRALayer(rank=16, base_inertia=0.7),
            'agent': DynamicLoRALayer(rank=8, base_inertia=0.5),
            'context': DynamicLoRALayer(rank=4, base_inertia=0.1)
        }
        self.migration_queues = {}  # per channel
        self.health_monitor = HealthMonitor()

    def snapshot(self) -> LoRASnapshot:
        """Create frozen copy for inference"""

    def forward(self, x, snapshot):
        """Apply all layers in sequence"""

    def migrate_circuit(self, circuit, from_layer, to_layer):
        """Handle upward migration with bandwidth check"""

    def system_tick(self):
        """Per-step maintenance: entropy budget, migration queue processing"""
```

**Shared vs Individual:**
- `Foundation`, `Domain`, `Swarm`: Shared across agents (single instance, synchronized access)
- `Agent`, `Context`: Individual per agent (each agent has its own instance)

---

### 2. `DynamicLoRALayer` (Individual Layer)

A single LoRA layer with emergent temporal dynamics.

**State:**
```python
class DynamicLoRALayer:
    # LoRA matrices
    A: Tensor  # (base_dim, rank)
    B: Tensor  # (rank, base_dim)

    # Baselines (for drift computation)
    A_init: Tensor
    B_init: Tensor

    # Emergent properties
    base_inertia: float              # configured
    interaction_history: deque       # recent activations
    screening_threshold: float       # adaptive
    entropy_budget: (float, float)   # (min, max)

    # Per-parameter state
    phase_state: Tensor[bool]  # SCREENED or FREE for each param
    activity_mask: Tensor      # recent activation intensity
```

**Key Methods:**
```python
def forward(self, x):
    """Apply LoRA transformation, record activation"""

def compute_effective_inertia(self):
    """Emergent temporal scale from interaction density"""

def update(self, gradient, strength=1.0):
    """Apply gradient with dynamic inertia and phase-based decay"""

def maintain_homeostasis(self):
    """Adjust screening_threshold if entropy out of bounds"""

def consolidate_circuit(self, circuit):
    """Strengthen a specific activation pattern"""

def compute_activity_mask(self):
    """Which parameters recently activated (for screening)"""
```

**Phase-Based Dynamics:**
```python
def update(self, gradient, strength):
    # Compute effective inertia from interaction density
    inertia = self.compute_effective_inertia()
    learning_rate = 1.0 - inertia

    # Apply gradient
    self.A += learning_rate * strength * gradient.A
    self.B += learning_rate * strength * gradient.B

    # Phase-based decay
    activity_mask = self.compute_activity_mask()

    for param in [self.A, self.B]:
        if activity_mask[param] > screening_threshold:
            # SCREENED: locked by interactions
            decay_rate = 0.0001
        else:
            # FREE: drift to base
            decay_rate = 0.01

        param = param * (1 - decay_rate) + param_init * decay_rate
```

---

### 3. `CircuitDetector` (Pattern Recognition)

Identifies stable activation patterns (circuits) from trajectories.

**Integration with WaveGen:**
```python
class CircuitDetector:
    def __init__(self, wavegen_cache):
        self.wavegen = wavegen_cache
        self.autocorr_window = 100
        self.stability_threshold = 0.7

    def detect_from_trajectory(self, trajectory):
        """Extract circuit from WaveGen trajectory"""
        # Trajectory = sequence of (token, logits, activations)
        activation_pattern = self.extract_pattern(trajectory)

        # Check if pattern is stable
        autocorr = self.compute_autocorrelation(activation_pattern)

        if autocorr > self.stability_threshold:
            return Circuit(
                nodes=activation_pattern.active_neurons,
                edges=activation_pattern.strong_connections,
                activation_signature=activation_pattern.hash()
            )
        return None

    def compute_autocorrelation(self, pattern):
        """Check if pattern repeats over time"""
        history = self.pattern_history[-self.autocorr_window:]
        return correlation(pattern.signature, history)

    def monitor_cache_hits(self):
        """Translate WaveGen cache stats to circuit consolidation signals"""
        for entry in self.wavegen.cache:
            if entry.hit_rate > consolidation_threshold:
                circuit = self.extract_circuit(entry.trajectory)
                yield (circuit, entry.hit_rate)
```

**Output:** `Circuit` objects ready for migration.

---

### 4. `MigrationQueue` (Bandwidth Management)

Manages upward migration with bandwidth limits and priority.

**Per-Channel Queue:**
```python
class MigrationQueue:
    def __init__(self, from_layer, to_layer, bandwidth):
        self.from_layer = from_layer
        self.to_layer = to_layer
        self.bandwidth = bandwidth  # fraction of params per tick
        self.pending = []  # priority queue (autocorr-sorted)
        self.current_transfer = 0.0

    def enqueue_migration(self, circuit):
        """Add circuit to migration queue"""
        circuit_size = circuit.parameter_count / self.from_layer.total_params

        # Emergency bypass for critical patterns
        if circuit.criticality > EMERGENCY_THRESHOLD:
            self.execute_migration(circuit, self.to_layer)
            return

        # Check bandwidth
        if self.current_transfer + circuit_size <= self.bandwidth:
            self.execute_migration(circuit, self.to_layer)
            self.current_transfer += circuit_size
        else:
            # Queue with priority
            self.pending.append(circuit)
            self.pending.sort(key=lambda c: c.autocorr, reverse=True)

    def tick(self):
        """Per-step: decay current_transfer, process queue"""
        self.current_transfer *= 0.9  # 10% bandwidth restored

        while self.pending and self.current_transfer < self.bandwidth:
            circuit = self.pending.pop(0)
            self.enqueue_migration(circuit)  # try again
```

**Global Coordinator:**
```python
class MigrationCoordinator:
    def __init__(self):
        self.queues = {
            'context→agent': MigrationQueue(..., bandwidth=0.01),
            'agent→swarm': MigrationQueue(..., bandwidth=0.005),
            'swarm→domain': MigrationQueue(..., bandwidth=0.001),
            'domain→foundation': MigrationQueue(..., bandwidth=0.0001)
        }

    def migrate_circuit(self, circuit, from_layer, to_layer):
        channel_key = f'{from_layer}→{to_layer}'
        self.queues[channel_key].enqueue_migration(circuit)

    def tick_all(self):
        for queue in self.queues.values():
            queue.tick()
```

---

### 5. `ResonanceEngine` (Gradient Composition)

Applies resonance filtering to gradient updates.

**Horizontal Resonance (Swarm Layer):**
```python
class HorizontalResonance:
    def __init__(self, alpha=0.5, beta=-0.3):
        self.alpha = alpha  # amplification for aligned gradients
        self.beta = beta    # dampening for conflicting gradients
        self.current_direction = None

    def apply(self, new_gradient, swarm_gradient_accumulator):
        if self.current_direction is None:
            self.current_direction = normalize(swarm_gradient_accumulator)

        alignment = dot(new_gradient, self.current_direction)

        if alignment > 0:
            resonance_factor = 1 + self.alpha * alignment
        else:
            resonance_factor = 1 + self.beta * alignment

        weighted_gradient = new_gradient * resonance_factor
        return weighted_gradient
```

**Vertical Resonance (Cross-Layer):**
```python
class VerticalResonance:
    def __init__(self, gamma=0.3, delta=0.2, max_amplification=2.0):
        self.gamma = gamma  # amplification coefficient
        self.delta = delta  # dampening coefficient
        self.max_amp = max_amplification  # safety cap

    def apply(self, gradients_dict):
        """gradients_dict = {'context': g_c, 'agent': g_a, 'swarm': g_s}"""

        # Compute vertical alignment
        alignments = [
            dot(gradients_dict['context'], gradients_dict['agent']),
            dot(gradients_dict['agent'], gradients_dict['swarm']),
            dot(gradients_dict['context'], gradients_dict['swarm'])
        ]
        vertical_alignment = mean(alignments)

        if vertical_alignment > VERTICAL_THRESHOLD:
            # Constructive interference (universal pattern)
            amplification = min(
                1 + self.gamma * vertical_alignment,
                self.max_amp  # cap to prevent exponential explosion
            )
            for key in gradients_dict:
                gradients_dict[key] *= amplification

            # Signal for fast-track migration
            return gradients_dict, 'FAST_TRACK'

        elif vertical_alignment < -CONFLICT_THRESHOLD:
            # Destructive interference (conflict between layers)
            # Trust slow layers, dampen fast layers
            gradients_dict['context'] *= (1 - self.delta * abs(vertical_alignment))
            gradients_dict['agent'] *= (1 - self.delta/2 * abs(vertical_alignment))
            # swarm untouched (trust the collective)

            return gradients_dict, 'CONFLICT'

        else:
            # Neutral
            return gradients_dict, 'NORMAL'
```

---

### 6. `HealthMonitor` (System Diagnostics)

Tracks system health metrics and triggers corrective actions.

**Metrics:**
```python
class HealthMonitor:
    def check_entropy_health(self, layer):
        entropy_ratio = layer.current_entropy / layer.optimal_entropy()

        if 0.8 < entropy_ratio < 1.2:
            return 'HEALTHY'
        elif entropy_ratio < 0.5:
            return 'CRYSTALLIZATION_RISK'
        elif entropy_ratio > 2.0:
            return 'AMNESIA_RISK'
        else:
            return 'ACCEPTABLE'

    def check_bandwidth_health(self, migration_coordinator):
        metrics = {}
        for channel, queue in migration_coordinator.queues.items():
            metrics[channel] = {
                'queue_depth': len(queue.pending),
                'avg_wait_time': queue.average_wait_time(),
                'rejection_rate': queue.rejections / queue.total_requests
            }

        healthy = all(
            m['queue_depth'] < 10 and
            m['avg_wait_time'] < 5 and
            m['rejection_rate'] < 0.01
            for m in metrics.values()
        )
        return 'HEALTHY' if healthy else 'DEGRADED'

    def check_resonance_health(self, resonance_engine):
        alignments = resonance_engine.collect_vertical_alignments()
        mean_align = mean(alignments)

        fast_track_rate = resonance_engine.fast_track_count / resonance_engine.total_migrations
        conflict_rate = resonance_engine.conflict_count / resonance_engine.total_operations

        healthy = (
            abs(mean_align) > 0.1 and  # signal exists
            0.05 < fast_track_rate < 0.20 and
            conflict_rate < 0.05
        )
        return 'HEALTHY' if healthy else 'DEGRADED'

    def overall_health(self, system):
        return {
            'entropy': [self.check_entropy_health(l) for l in system.layers],
            'bandwidth': self.check_bandwidth_health(system.migration_coordinator),
            'resonance': self.check_resonance_health(system.resonance_engine),
            'timestamp': current_time()
        }
```

---

### 7. Integration Layer: `LoRABackend` (LLM Interface)

Extends `BaseLLMBackend` (from WaveGen) to support LoRA composition.

```python
class LoRABackend(BaseLLMBackend):
    def __init__(self, base_model, lora_hierarchy):
        super().__init__(base_model)
        self.lora_hierarchy = lora_hierarchy

    def generate_step(self, context, lora_snapshot):
        """Single generation step with LoRA composition"""
        # Tokenize
        input_ids = self.tokenize(context)

        # Forward through base
        hidden = self.base_model.embed(input_ids)

        # Apply LoRA stack (from snapshot)
        for layer_name in ['foundation', 'domain', 'swarm', 'agent', 'context']:
            lora_layer = lora_snapshot[layer_name]
            hidden = lora_layer.forward(hidden)

        # Final projection
        logits = self.base_model.lm_head(hidden)

        return logits

    def compute_gradient(self, loss, lora_snapshot):
        """Backprop through LoRA stack"""
        # Standard backprop, but only LoRA params get gradients
        # (base model frozen)
        gradients = {}
        for layer_name, layer in lora_snapshot.items():
            gradients[layer_name] = layer.compute_gradient(loss)
        return gradients
```

---

## System Execution Flow

### Per-Tick Coordination

```python
def system_tick():
    """Called after each inference or periodically"""

    # 1. Entropy Budget (homeostasis first)
    for layer in hierarchy.layers.values():
        layer.maintain_homeostasis()

    # 2. Collect gradients from all agents
    gradients = collect_all_gradients()

    # 3. Apply vertical resonance
    gradients, resonance_signal = resonance_engine.apply_vertical(gradients)

    # 4. Apply gradients to layers
    for layer_name, grad in gradients.items():
        hierarchy.layers[layer_name].update(grad)

    # 5. Circuit detection (from WaveGen cache)
    for circuit, hit_rate in circuit_detector.monitor_cache_hits():
        # Determine target layer based on hit_rate
        if hit_rate > domain_threshold:
            target = 'domain'
        elif hit_rate > swarm_threshold:
            target = 'swarm'
        else:
            target = 'agent'

        # Enqueue migration
        migration_coordinator.migrate_circuit(circuit, from_layer='context', to_layer=target)

    # 6. Process migration queues
    migration_coordinator.tick_all()

    # 7. Health check
    health = health_monitor.overall_health(hierarchy)
    if any(status == 'DEGRADED' for status in health.values()):
        log_warning(health)
```

---

## Architectural Variants

### Variant A: Snapshot Inference (Simpler, Recommended for v1.0)

**Inference:**
```python
snapshot = hierarchy.snapshot()  # freeze LoRA state

with frozen(snapshot):
    output = generate(prompt, snapshot)

# Updates happen AFTER inference
system_tick()
```

**Pros:**
- Deterministic (same input → same output)
- No race conditions
- WaveGen integration straightforward

**Cons:**
- LoRA updates delayed (between requests, not during)

---

### Variant B: Live Inference (Advanced, Experimental)

**Inference:**
```python
# No snapshot—LoRA can update mid-generation
for step in generation_loop(prompt):
    # Dynamic weight composition
    W_effective = base_weights + sum(lora.compute_delta() for lora in active_loras)

    logits = forward(step, W_effective)
    token = sample(logits)

    # Immediate learning
    if coherence(recent_trajectory) > threshold:
        lora_context.consolidate_pattern(recent_trajectory)
```

**Pros:**
- Immediate adaptation (learning during generation)
- More "biological" (brain updates while thinking)

**Cons:**
- Nondeterministic (weights change mid-inference)
- Complex (resonance conflicts, WaveGen cache invalidation)
- Requires careful synchronization

**Recommendation:** Start with Variant A. Experiment with B in research phase.

---

## Summary: Component Relationships

```
LoRAHierarchy (coordinator)
    ├── DynamicLoRALayer × 5 (foundation, domain, swarm, agent, context)
    ├── MigrationCoordinator
    │   └── MigrationQueue × 4 (one per channel)
    ├── ResonanceEngine
    │   ├── HorizontalResonance
    │   └── VerticalResonance
    ├── CircuitDetector
    │   └── Integrates with WaveGen cache
    └── HealthMonitor

LoRABackend
    └── Wraps base model + LoRA composition
```

**WaveGen sits alongside:**
- Observes trajectories
- Provides circuit candidates to CircuitDetector
- Evicts patterns once consolidated in LoRA
