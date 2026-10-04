The per-light design changes a few things, so here's the updated plan.

## The core design decisions

**The signals.** 4 approaches (N, E, S, W) × 4 lanes (left, forward, forward, right) gives **16 lane signals**, each on or off. Each lane has one arrow, and "light on" means cars from that lane's queue can discharge toward that movement.

**The conflict matrix.** This is the heart of the project. It's a 16×16 table saying which pairs of lanes can't be green together, because their paths cross. Examples:
- North forward and east forward conflict (perpendicular).
- North forward and south forward are compatible.
- A left turn conflicts with the opposing forward lanes (protected left).
- Right turns are mostly safe with the opposing approach's movements.

Treat protected vs permissive turns as a dial. Define the matrix once, in a config, from the geometry, and everything else (UI validation, models, tests) reads from it.

**What the model outputs.** There are two ways to design the action space, and comparing them is a good experiment:

- **A. Pick a valid configuration.** Enumerate every legal combination of lights up front (the maximal sets of mutually compatible signals). There will be a few dozen. The model picks one, so it can never choose something unsafe.
- **B. Output 16 on/off values directly**, with a safety layer on top. The layer is either a mask that removes conflicting options, a repair step that drops the lowest-priority conflicting light, or a penalty for conflicts. This is a much bigger action space and harder to learn, but more flexible.

Start with A and add B as a variation.

**Is scikit-learn enough?** Not for the decision-making. scikit-learn models predict from labelled examples, but nobody labels "the best lights to turn on". The agent has to discover that through reward, which is reinforcement learning (PyTorch). scikit-learn still has two uses here:
- **Imitation learning:** train a classifier to copy the max-pressure controller's choices. Either predict the configuration ID or use multi-output classification for the 16 lights.
- **Forecasting:** predict queue lengths or arrivals as an extra input feature.

## The UI

Keep it dead simple, and keep rendering separate from the simulation: the draw function takes a simulation state and shows it. That separation is what makes side-by-side comparison easy later (two sims, same seed, two draw calls).

```
                 NORTH approach
            L1   F2   F3   R4        (arrow lit if green)
           [ 4] [ 7] [ 2] [ 0]       <- queue per lane
 WEST  ...                                   ... EAST
                 SOUTH approach
```

- Under each lane: its label (e.g. `N-L`, `N-F1`), its queue count, and its light state.
- On each exit edge: total cars that have passed through.
- At the top: total cars moved, average wait, and the current controller's name.
- Controls: step, play/pause, speed, and manual toggles for the 16 lights (manual mode is how you test the conflict rules).
- Tooling: Streamlit is fine for the dashboard and parameter sliders. For smooth live animation, pygame or tkinter is easier than Streamlit's rerun model.

## Modules (build them in this order)

**1. Intersection definition and conflict matrix**
- Lane names, movements, and the compatibility rules in one config.
- Verify: write the matrix from the geometry on paper first, then compare it with the code. Test that it's symmetric.
- Dials: protected vs permissive left turns, right-on-red style rules.

**2. Simulator**
- Cars arrive per lane (Poisson, set by the arrival rate). When a lane is green, up to N cars per step discharge toward the exit edge. Counters track per-edge and total throughput.
- Verify with tests: no conflicting lights are ever on at once (run it with a random controller for 100k steps), and cars are conserved (arrived = queued + passed).
- Dials: arrival rates per lane, discharge rate, step length, minimum green, yellow/all-red clearance, turn-split percentages.

**3. UI**
- Everything in the layout above, with manual light control first, then a "run a controller" mode.
- Verify: toggle two conflicting lights and confirm the UI rejects it.

**4. Baseline controllers**
- **Random valid configuration** every k steps. This is the floor, and any learned model must beat it.
- **Fixed cycle** through a preset list of configurations.
- **Actuated:** give green to a lane or phase when its queue passes a threshold.
- **Max-pressure** (per movement): choose the valid configuration with the highest total pressure.
- Verify: max-pressure should clearly beat random. If it doesn't, suspect the simulator or the conflict matrix.

**5. Evaluation harness**
- Fixed scenarios (light, rush hour, one-direction heavy), a fixed set of seeds, and a standard metrics table.
- Verify: same seed plus same controller gives identical results.

**6. Experiment config and tracking**
- One YAML per run. Example:
```yaml
controller: dqn
action_space: configs        # or: multi_binary_masked
state: [queues, waits, current_config, time_in_config]
reward: neg_total_wait
network: {type: mlp, hidden: [128, 128], activation: relu}
optimizer: {name: adam, lr: 0.0003}
training: {steps: 200000, batch: 64, gamma: 0.99, seeds: [0,1,2,3,4]}
scenario: rush_hour
```
- Results go to TensorBoard plus your markdown table.

**7. scikit-learn track**
- Imitation of max-pressure (random forest, gradient boosting, logistic regression).
- Queue forecasting as an added input feature.
- Dials: features, model family, cross-validation, dataset size.

**8. RL track**
- Tabular Q-learning on a heavily simplified state (bucketed queues per approach).
- DQN over valid configurations, then Double and Dueling variants.
- DQN or PPO with the 16-light output plus masking.
- Dials: reward, state, network, optimizer, exploration, replay buffer, gamma.

**9. Jev as a controller**
- Send the state as JSON and declare the answer as "pick one of these configuration IDs".
- Handle its latency and add a fallback to max-pressure when confidence is low.
- Optionally distill its choices into a small local model.

**10. Side-by-side comparison view**
- Two sims with the same arrivals, two controllers, one screen.

## Things to watch

- **Switching cost:** without clearance time (yellow/all-red) or a switch penalty, agents flicker lights every step. Add that constraint in the simulator, and try a switch penalty in the reward as a variation.
- **Starvation:** an agent may ignore a low-traffic lane forever. Add a max-wait metric alongside average wait, and perhaps a fairness term in the reward.
- **Invalid actions:** with the 16-light output you need an explicit policy (mask, repair, or penalty), and that choice is an experiment in itself.
- **Right turns:** they conflict with pedestrians in real life. Ignoring pedestrians is fine for v1, but write the assumption down.

## How you'd work with this

For each module, tell the AI the goal, the constraint, and the dials in plain language, for example: "Implement module 4, max-pressure at the movement level, using the compatibility matrix from the config, picking among enumerated valid configurations." Then read the code, run the module's verification checks, and log the result in your markdown file.

Want me to put this plan into a markdown file you can keep, and draft the conflict matrix for the 16 lanes as a starting point?