# System Blueprint: 03 - Data Flow & Interaction Lifecycle

This document describes the end-to-end lifecycle of inference, learning, and consolidation in the DynamicLoRA system.

## Scenario: Single-Agent Inference with Learning

### Initial State
- `LoRAHierarchy` initialized with all layers
- `Foundation`, `Domain` partially populated (from prior training)
- `Swarm` empty (no team patterns yet)
- `Agent` empty (new agent)
- `Context` empty (fresh conversation)
- `WaveGen` cache empty

---

## Lifecycle Steps

### 1. **User Request Arrives**

```
User → System: "Write a function to parse JSON in Python"
```

**System actions:**
- Create `LoRASnapshot` (freeze current state of all layers)
- Initialize `WaveGenCache` for this session
- Prepare `CircuitDetector`

---

### 2. **Inference Begins (Snapshot Mode)**

```python
snapshot = hierarchy.snapshot()  # frozen LoRA state

with frozen(snapshot):
    trajectory = []
    context = tokenize(prompt)

    for step in range(max_tokens):
        # Forward pass through LoRA stack
        hidden = base_model.embed(context)
        hidden = snapshot['foundation'].forward(hidden)
        hidden = snapshot['domain'].forward(hidden)
        hidden = snapshot['swarm'].forward(hidden)
        hidden = snapshot['agent'].forward(hidden)
        hidden = snapshot['context'].forward(hidden)

        logits = base_model.lm_head(hidden)
        token = sample(logits)

        # Record for WaveGen
        trajectory.append(StepObservation(
            token_id=token,
            logits=logits,
            activations=hidden,
            step_index=step
        ))

        context = append(context, token)

        # WaveGen analysis (coherence tracking)
        if step >= window_size:
            coherence = wavegen.compute_coherence(trajectory[-window_size:])

            if coherence > jump_threshold:
                # Potential cache hit (WaveGen JUMP)
                # But first inference—cache empty, skip

    output = detokenize(context)
```

**Result:** Generated code:
```python
import json

def parse_json(json_string):
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error: {e}")
        return None
```

---

### 3. **Post-Inference: WaveGen Caching**

```python
# Analyze full trajectory
coherence = wavegen.compute_coherence(trajectory)

if coherence > cache_threshold:
    # Stable trajectory—cache it
    signature = wavegen.create_signature(
        trajectory=trajectory,
        provenance=snapshot.provenance_hash
    )

    wavegen.cache.store(
        signature=signature,
        block=GeneratedBlock(
            text=output,
            token_ids=context,
            metadata={'coherence': coherence}
        )
    )
```

**WaveGen cache now contains:**
```
Entry #1:
  Signature: hash("parse JSON" + snapshot_hash)
  Block: "import json\ndef parse_json..."
  Stats: uses=1, hits=0, successes=0
```

---

### 4. **Post-Inference: LoRA Update (Context Layer)**

```python
# Evaluate quality
quality = evaluate_output(output, task)

if quality > threshold:
    # Good result—strengthen patterns in Context layer
    circuit = circuit_detector.extract_from_trajectory(trajectory)

    # Update Context LoRA (fast layer)
    lora_context.consolidate_circuit(circuit, strength=0.1)
```

**LoRA_context now contains:**
- Weak circuit for "import json → json.loads → error handling"
- Screening state: mostly FREE (first use)

---

### 5. **Second Request: Same Task**

```
User → System: "Write another JSON parser"
```

**Inference:**

```python
snapshot = hierarchy.snapshot()  # includes updated Context

# Forward pass (same as before)
trajectory = []
for step in generation_loop(prompt, snapshot):
    # ...
    coherence = wavegen.compute_coherence(trajectory[-window_size:])

    if coherence > jump_threshold:
        # Check WaveGen cache
        signature = wavegen.create_signature(trajectory[-window_size:], snapshot.hash)

        if cached_block := wavegen.cache.lookup(signature):
            # CACHE HIT!
            # Insert cached block, terminate generation
            output = cached_block.text
            wavegen.cache.record_hit(signature)
            break  # skip token-by-token generation

# If no cache hit, continue generation
```

**Result:** Cache hit! Output identical, but generated 10x faster (WaveGen JUMP).

**WaveGen cache updated:**
```
Entry #1:
  uses=2, hits=1, successes=1
```

---

### 6. **Third, Fourth, Fifth Requests: Pattern Stabilizes**

Over multiple requests:
- WaveGen cache hit rate increases (hits=10, uses=12)
- `LoRA_context` circuit gets repeatedly activated
  - Activity mask increases
  - Phase transition: FREE → SCREENED (circuit now locked)
- Autocorrelation detector notices: "This circuit appears repeatedly"

```python
# CircuitDetector monitoring
autocorr = circuit_detector.compute_autocorrelation(circuit, window=100)

if autocorr > stability_threshold:
    # Pattern is stable!
    # Candidate for migration Context → Agent
    migration_coordinator.enqueue_migration(
        circuit=circuit,
        from_layer='context',
        to_layer='agent'
    )
```

---

### 7. **Migration: Context → Agent**

**MigrationQueue processing:**

```python
# Migration queue for Context→Agent
queue = migration_coordinator.queues['context→agent']

# Check bandwidth (1% of params per tick)
circuit_size = circuit.parameter_count / lora_context.total_params

if queue.current_transfer + circuit_size <= queue.bandwidth:
    # Execute migration
    lora_agent.consolidate_circuit(circuit, strength=0.5)
    lora_context.weaken_circuit(circuit, strength=0.3)  # gradual handoff

    # WaveGen: evict cache entry (now internalized)
    wavegen.cache.evict(signature)

    queue.current_transfer += circuit_size
else:
    # Queue for next tick
    queue.pending.append(circuit)
```

**Result:**
- Circuit migrated from Context → Agent
- LoRA_agent now contains "JSON parsing" circuit
- WaveGen cache freed (space for new patterns)
- Context layer freed (can learn new task-specific patterns)

---

### 8. **Multi-Agent: Swarm Consolidation**

**Scenario:** Another agent (Agent B) also does JSON parsing.

```python
# Agent A's LoRA_agent has JSON circuit
# Agent B generates JSON code independently

# Both agents update shared LoRA_swarm
gradient_a = agent_a.compute_gradient(json_task)
gradient_b = agent_b.compute_gradient(json_task)

# Horizontal Resonance
alignment = dot(gradient_a, gradient_b)

if alignment > 0:
    # Constructive interference (both learned same pattern)
    resonance_factor = 1 + alpha * alignment
    combined_gradient = (gradient_a + gradient_b) * resonance_factor
else:
    # Destructive interference (conflicting patterns)
    resonance_factor = 1 + beta * alignment  # beta < 0
    combined_gradient = (gradient_a + gradient_b) * resonance_factor

# Update Swarm
lora_swarm.update(combined_gradient)
```

**Result:**
- Swarm layer now contains shared "JSON parsing" circuit
- Future agents can inherit this pattern immediately
- No explicit message passing required

---

### 9. **Vertical Resonance: Fast-Track to Domain**

**Scenario:** Circuit is useful at all levels (Context, Agent, Swarm).

```python
# Check vertical alignment
gradients = {
    'context': grad_context,
    'agent': grad_agent,
    'swarm': grad_swarm
}

vertical_alignment = mean([
    dot(gradients['context'], gradients['agent']),
    dot(gradients['agent'], gradients['swarm']),
    dot(gradients['context'], gradients['swarm'])
])

if vertical_alignment > VERTICAL_THRESHOLD:
    # Universal pattern!
    # Amplify across all layers
    for layer in gradients.keys():
        gradients[layer] *= (1 + gamma * vertical_alignment)

    # Fast-track migration to Domain (bypass normal queue)
    migration_coordinator.fast_track_migration(
        circuit=circuit,
        target='domain'
    )
```

**Result:**
- Circuit jumps Context → Domain (skipping Agent/Swarm intermediate steps)
- Domain layer now contains fundamental "JSON parsing" knowledge
- This becomes available to ALL agents in ALL domains (not just this team)

---

### 10. **Entropy Budget Triggers Homeostasis**

**Scenario:** After many migrations, Domain layer becomes too rigid.

```python
# HealthMonitor check
current_entropy = lora_domain.compute_entropy()
optimal_entropy = lora_domain.optimal_entropy()

entropy_ratio = current_entropy / optimal_entropy

if entropy_ratio < 0.5:
    # CRYSTALLIZATION RISK
    # Too many parameters SCREENED, system too rigid

    # Corrective action: raise screening threshold
    lora_domain.screening_threshold *= 1.1

    # Result: some parameters transition SCREENED → FREE
    # Entropy increases, balance restored
```

**Emergent property:** System self-regulates without manual intervention.

---

### 11. **Bandwidth Limit Prevents Avalanche**

**Scenario:** 50 circuits become stable simultaneously.

```python
# CircuitDetector finds 50 stable circuits
for circuit in stable_circuits:
    migration_coordinator.enqueue_migration(circuit, 'context', 'agent')

# MigrationQueue
queue = migration_coordinator.queues['context→agent']

# Bandwidth = 1% per tick
# 50 circuits would require 20% → exceeds limit

# Result:
#   - Top 5 circuits (highest autocorr) migrate immediately
#   - Remaining 45 queued (priority = autocorr strength)
#   - Over next 10 ticks, queue gradually processes

# Domain layer protected from sudden overload
```

---

## Complete Lifecycle Summary

```
┌─────────────────────────────────────────────────────────────┐
│ 1. User Request                                             │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌────────────────▼────────────────────────────────────────────┐
│ 2. Snapshot LoRA Hierarchy (frozen for inference)           │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌────────────────▼────────────────────────────────────────────┐
│ 3. Forward Pass (Base + Foundation + Domain + Swarm +       │
│                  Agent + Context)                           │
│    - WaveGen observes trajectory                            │
│    - Check cache (JUMP if hit)                              │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌────────────────▼────────────────────────────────────────────┐
│ 4. Post-Inference: WaveGen Cache                            │
│    - If coherent → store trajectory                         │
│    - Track hit rate                                         │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌────────────────▼────────────────────────────────────────────┐
│ 5. Post-Inference: LoRA Update                              │
│    - Extract circuit from trajectory                        │
│    - Update Context layer                                   │
│    - Phase transition (FREE → SCREENED if repeated)         │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌────────────────▼────────────────────────────────────────────┐
│ 6. Pattern Stabilization (over multiple requests)           │
│    - Autocorrelation detector notices repetition            │
│    - Circuit becomes migration candidate                    │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌────────────────▼────────────────────────────────────────────┐
│ 7. Migration Queue Processing                               │
│    - Check bandwidth                                        │
│    - Migrate if available                                   │
│    - Else: queue with priority                              │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌────────────────▼────────────────────────────────────────────┐
│ 8. Upward Migration (Context → Agent → Swarm → Domain)      │
│    - Consolidate in target layer                            │
│    - Weaken in source layer                                 │
│    - Evict from WaveGen cache                               │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌────────────────▼────────────────────────────────────────────┐
│ 9. Resonance Amplification                                  │
│    - Horizontal: multi-agent consensus                      │
│    - Vertical: universal patterns fast-tracked              │
└────────────────┬────────────────────────────────────────────┘
                 ↓
┌────────────────▼────────────────────────────────────────────┐
│ 10. Health Monitoring & Homeostasis                         │
│     - Entropy budget check                                  │
│     - Bandwidth queue health                                │
│     - Resonance quality                                     │
│     - Corrective actions if needed                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Observations

### Separation of Concerns

**WaveGen (Inference Optimization):**
- Fast cache-based generation
- Works in **trajectory space** (sequences of tokens)
- External to model parameters

**DynamicLoRA (Learning & Consolidation):**
- Continuous adaptation
- Works in **parameter space** (model weights)
- Internal model structure

**Integration:**
- WaveGen cache hit rate → signal for LoRA consolidation
- LoRA snapshot hash → WaveGen cache key (provenance)
- Circuit extraction: WaveGen trajectories → LoRA patterns

### Emergent Properties

1. **Temporal scales not configured, but emerge:**
   - Foundation changes rarely (low interaction density)
   - Context changes frequently (high interaction density)

2. **No manual triggers for migration:**
   - Autocorrelation detects stability
   - Bandwidth queues prevent overload
   - Vertical resonance fast-tracks universal patterns

3. **Self-healing through homeostasis:**
   - Entropy budget prevents crystallization/amnesia
   - Resonance filtering rejects bad patterns
   - Health monitoring triggers corrections

### Multi-Agent Coordination

**No central orchestrator:**
- Each agent updates shared layers independently
- Resonance filtering ensures consensus
- Conflicting patterns naturally cancel out

**Implicit communication:**
- Agents "feel" each other's contributions through parameter space
- Like gravity: not messages, but curvature

---

## Comparison to Traditional Approaches

| Aspect | Traditional Fine-Tuning | DynamicLoRA |
|--------|------------------------|-------------|
| **When learning happens** | Offline, batch | Online, continuous |
| **Catastrophic forgetting** | Yes (new data overwrites old) | No (hierarchy + phase transitions) |
| **Multi-agent learning** | Requires explicit aggregation | Emergent through resonance |
| **Temporal structure** | None (all weights same speed) | Emergent hierarchy (fast/slow layers) |
| **Interpretability** | Black box | Circuit-based, traceable |
| **Cache integration** | N/A | WaveGen signals consolidation |
| **Stability** | Requires careful hyperparameter tuning | Self-regulating (homeostasis) |
