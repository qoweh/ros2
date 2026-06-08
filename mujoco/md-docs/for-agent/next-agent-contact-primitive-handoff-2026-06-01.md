# next agent contact primitive handoff

## 1. project purpose

This workspace contains several projects, but the active project is `pingpong_rl2`.

Its goal is not generic table-tennis rally play. The current target is much narrower and more precise:

- build a MuJoCo + Gymnasium environment where a Franka-mounted racket can perform repeated ping-pong keep-up
- find the minimum control abstraction that makes repeated `3+` useful bounces possible under a narrow reset regime
- only after that structural gate passes, resume PPO training in a serious way

The important mindset is:

- do not treat this as a reward-tuning problem first
- do not treat this as a “train longer” PPO problem first
- first prove whether the environment and control abstraction can even express the target behavior

`pingpong_rl2` exists separately from `pingpong_rl` because the original branch had too much accumulated baggage. This project intentionally re-checks the keep-up problem with a smaller, more auditable stack.

## 2. current headline status

As of 2026-06-01, the work is not finished, but the main scientific uncertainty is mostly resolved.

Resolved facts:

- geometry is not the ceiling
- the current plain `position_strike` abstraction is a ceiling
- tilt-only residual primitive is not enough
- phase-aware non-tilt residuals do help and can stably open `2+`
- even with the best current scripted candidate, the system still tops out at `max_useful_bounces = 2`
- simple local next branches have now also been tested and falsified:
  - follow-up contact offset refinement
  - explicit follow-up lift residual
  - constant strike x residual
  - linear state-dependent XY correction gain
  - simple late strike-only z pulse

Current best scripted candidate:

- `action_mode = position_strike_tilt`
- `strike_position_residual = (0.0, 0.0, +0.01)`
- `strike_tilt_residual = (0.0, -0.02)`
- `followup_strike_lift_boost = 0.02`
- base env contract:
  - `strike_z_boost = 0.024`
  - `strike_tilt_ramp_pitch = -0.06`
  - `followup_strike_target_tilt = (-0.06, 0.0)`
  - `post_contact_return_assist_weight = 0.5`
  - `post_contact_return_max_intercept_time = 0.6`

Best 100-episode confirmation for that candidate:

- `mean_useful_bounces = 0.58`
- `max_useful_bounces = 2`
- `one_or_more_useful_bounce_rate = 0.47`
- `two_or_more_useful_bounce_rate = 0.11`
- `three_or_more_useful_bounce_rate = 0.0`

That is the most important single number block right now.

## 3. what has been proved, in order

### stage 1: baseline and trace sanity

Reports:

- `14_contact_trace_sanity_report.md`
- `15_contact_feasibility_map_report.md`

What was learned:

- contact metrics must prefer stabilized post-contact outgoing velocity, not the first contact substep
- the best plain scripted `position_strike` candidate still fails the `3+` gate
- the ceiling under the current abstraction was about `max = 2`

### stage 2: upper bound diagnosis

Report:

- `16_contact_upper_bound_report.md`

What was learned:

- a diagnostic contact oracle that partially replaces outgoing velocity after contact was added
- this is diagnostic-only and must never become a training feature
- with oracle blend `0.5` to `0.75`, the environment opens stable `3+` and even time-limited repeated keep-up

This proved:

- the task geometry is not the bottleneck
- the remaining bottleneck is controller / action abstraction

### stage 3: tilt-only primitive branch

Reports:

- `17_contact_primitive_training_plan.md`
- `18_tilt_primitive_scripted_feasibility_report.md`

What was implemented:

- `action_mode = position_strike_tilt`
- two tilt residual action dimensions on top of the strike contract
- PPO preset `contact_primitive_candidate`

What was learned:

- constant tilt residuals failed
- strike-phase-only tilt residuals also failed
- best tilt-only candidate still had `max_useful_bounces = 1`

Conclusion:

- tilt-only residuals are not the missing structural axis

### stage 4: non-tilt residual branch

Report:

- `19_non_tilt_contact_residual_report.md`

What was implemented:

- heuristic diagnostic can now emit phase-aware position residuals
- heuristic diagnostic can also emit explicit follow-up-lift residuals
- new env mode `position_strike_tilt_lift`
- new PPO preset `contact_lift_candidate`

What was learned:

- adding `strike z = +0.01` on top of the best small strike roll residual is the first thing that reliably opens `2+`
- this signal survives 100-episode confirmation
- explicit lift residual, constant strike x, and follow-up offset do not push beyond the `max = 2` ceiling

### stage 5: cheap local falsification after that

Report:

- `20_state_dependent_contact_point_and_timing_report.md`

What was implemented:

- state-dependent strike XY correction gain in the heuristic
- strike-phase-only z pulse in the heuristic

What was learned:

- linear XY correction gain does nothing in the current narrow reset regime
- simple late strike-only z pulse is actually worse than the current best candidate

Conclusion:

- the cheap local branches are exhausted
- the next branch must be a richer state-dependent contact primitive, not another small scalar tweak

## 4. current code surfaces that matter most

### env core

- `pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py`

This file is the main behavior contract.

Important responsibilities:

- action parsing for all action modes
- strike / follow-up contract logic
- phase calculation: `prepare`, `strike`, `return_shaping`, `recovery`
- contact/outgoing metrics
- reward and success terms
- oracle diagnostic hook

Current important action modes:

- `position_strike`
- `position_strike_tilt`
- `position_strike_tilt_lift`

### heuristic diagnostics

- `pingpong_rl2/src/pingpong_rl2/controllers/heuristic_keepup.py`
- `pingpong_rl2/scripts/run_heuristic_keepup_diagnostic.py`

These two files are the fastest way to test a structural hypothesis before PPO.

If you are not sure whether a new primitive matters, do not start with PPO. Encode the hypothesis here first and run narrow scripted diagnostics.

### PPO entrypoint

- `pingpong_rl2/scripts/run_ppo_learning.py`

This file now supports:

- `contact_primitive_candidate`
- `contact_lift_candidate`
- heuristic bootstrap collection for `position_strike`, `position_strike_tilt`, and `position_strike_tilt_lift`

Important policy rule:

- do not use this as the main research loop until the scripted gate improves again

### naming / defaults

- `pingpong_rl2/src/pingpong_rl2/defaults.py`
- `pingpong_rl2/src/pingpong_rl2/utils/ppo_runs.py`

These contain run-name aliases and smoke naming for the newer action modes.

## 5. validated commands and environment

Environment assumptions:

- OS: macOS
- conda env: `mujoco_env`
- active project root: `.../mujoco/pingpong_rl2`
- all reliable runtime checks were executed with `PYTHONPATH=src`

Useful commands:

Run scripted diagnostic:

```bash
cd pingpong_rl2
PYTHONPATH=src conda run -n mujoco_env python scripts/run_heuristic_keepup_diagnostic.py --episodes 20 --analysis-name test_run
```

Resolve PPO preset without training:

```bash
cd pingpong_rl2
PYTHONPATH=src conda run -n mujoco_env python -c "import sys; sys.path.insert(0, 'scripts'); import run_ppo_learning as r; from pingpong_rl2.envs import PingPongKeepUpGymEnv; sys.argv=['run_ppo_learning.py','--preset','contact_lift_candidate']; args=r.parse_args(); preset=r.apply_env_preset(args); tilt=r.resolve_tilt_profile(args); env_kwargs=r.env_kwargs_from_args(args); env=PingPongKeepUpGymEnv(**env_kwargs); print({'preset': preset, 'tilt_profile': tilt, 'action_mode': env_kwargs['action_mode'], 'action_size': env.base_env.action_size, 'observation_size': env.base_env.observation_size}); env.close()"
```

Collect a 1-episode heuristic bootstrap sample for preset sanity:

```bash
cd pingpong_rl2
PYTHONPATH=src conda run -n mujoco_env python -c "import sys; sys.path.insert(0, 'scripts'); import run_ppo_learning as r; sys.argv=['run_ppo_learning.py','--preset','contact_lift_candidate']; args=r.parse_args(); r.apply_env_preset(args); env_kwargs=r.env_kwargs_from_args(args); data=r.collect_heuristic_bootstrap_dataset(env_kwargs=env_kwargs, episodes=1, seed=701, min_useful_bounces=0, max_samples=32, sample_mode='episode'); print({'action_mode': env_kwargs['action_mode'], 'obs_shape': tuple(data['observations'].shape), 'act_shape': tuple(data['actions'].shape)})"
```

## 6. what has already been tried and should not be repeated blindly

These are not theoretical guesses. They were already tested.

- more PPO on plain `position_strike`
- more reward shaping as the primary branch
- tilt-only constant residuals
- tilt-only strike-phase residuals
- follow-up contact offset alone
- explicit follow-up-lift residual around the current best candidate
- constant strike x residual around the current best candidate
- linear state-dependent XY gain using `anchor - predicted_intercept`
- simple late strike-only z pulse

Repeating these without a materially different structural hypothesis is low-value.

## 7. why the current conclusion is what it is

The key evidence chain is:

1. plain scripted feasibility map failed `3+`
2. contact oracle opened stable `3+`
3. therefore geometry is not the ceiling
4. tilt-only primitive still failed
5. phase-aware non-tilt residual finally opened `2+`
6. but even the best 100-episode candidate still capped at `max = 2`
7. nearby local branches after that also failed to move the ceiling

So the only defensible next step is not “search a little harder over the same residuals.”

It is to add a more expressive contact primitive.

## 8. recommended next branch

The strongest next branch is one of these two.

### option A: explicit state-dependent contact-point primitive

Idea:

- stop treating strike x/y as a constant residual or simple linear correction
- add an action semantic that directly parameterizes contact point relative to the predicted intercept frame
- likely better than another tilt or lift tweak

Examples:

- contact-frame `dx, dy`
- contact-frame `radial / tangential` offset
- offset only inside the final strike window, not in all `prepare` steps

### option B: richer impact-time primitive

Idea:

- stop treating z timing as a constant offset or late scalar pulse
- expose a primitive that changes impact timing only inside a short contact window or through a parameterized pulse profile

Examples:

- a two-parameter z pulse with onset threshold + magnitude
- a short strike-window vertical lead primitive
- a contact-window height target separate from the earlier prepare target

Between the two, option A currently looks slightly more promising conceptually, but the evidence is not decisive enough to ban option B.

## 9. suggested operating rules for the next agent

1. Start from `keepup_env.py` or `heuristic_keepup.py`, not from PPO.
2. Form one local falsifiable hypothesis and test it with scripted diagnostics first.
3. Use 30-episode narrow sweeps for local screening.
4. Only run 100-episode confirmation after a candidate actually improves the gate.
5. Do not reopen broad reward tuning unless a new primitive first shows promise.
6. Do not promote the oracle to a training feature.
7. Keep changes surgical; this repo now has enough signal that broad rewrites are more likely to destroy auditability than help.

## 10. current bottom line

The project is much closer to the end than it was before.

What remains is not “figure out what the problem is.”

We know what the problem is:

- the current abstraction can now produce stable `2+`
- but it still cannot express the contact behavior needed for stable `3+`

The next agent should spend its time on that missing expressive axis, not on rediscovering the upper-bound or re-running already-failed small ablations.
