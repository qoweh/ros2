# Next Agent 5.5 Keep-Up Gate And Primitive Instructions

## Context

Project root:

`/Users/pilt/project-collection/ros2/graduation-prj/pingpong_rl2`

Final goal:

Train a MuJoCo/Gymnasium RL policy where a Franka Panda robot arm holds a ping-pong racket and repeatedly hits the ball upward so that the next ball comes back into a hittable region.

Current issue:

The robot can sometimes contact the ball, but the ball is often sent away from the robot instead of being returned to a repeatable strike zone. Recent work has mostly changed reward weights, presets, and small residual values. That is not enough. The next work should focus on whether the action/control primitive can express the desired contact geometry.

Important reference files:

- `next-agent-contact-primitive-handoff-2026-06-01.md`
- `last-agent-answer.txt`
- `next-agent-spark-keepup-completion-plan-2026-06-01.md`
- `pingpong_rl2/docs/report/16_contact_upper_bound_report.md`
- `pingpong_rl2/docs/report/17_contact_primitive_training_plan.md`
- `pingpong_rl2/docs/report/19_non_tilt_contact_residual_report.md`
- `pingpong_rl2/docs/report/20_state_dependent_contact_point_and_timing_report.md`
- `pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py`
- `pingpong_rl2/src/pingpong_rl2/controllers/heuristic_keepup.py`
- `pingpong_rl2/src/pingpong_rl2/envs/pingpong_sim.py`
- `pingpong_rl2/scripts/run_heuristic_keepup_diagnostic.py`
- `pingpong_rl2/scripts/run_ppo_learning.py`

## Core Diagnosis

Do not treat this as a PPO duration or reward-weight problem first.

The existing contact oracle showed that the task geometry is feasible. If the post-contact velocity is guided toward a repeatable keep-up target, multiple useful bounces become possible. That means the environment is not fundamentally impossible.

The likely bottleneck is that the current action/control abstraction does not directly let the policy choose:

- where on the racket/ball contact should happen,
- what contact-frame direction should be produced,
- how impact timing should be adjusted,
- how racket normal and upward velocity should align at contact.

So the next agent should work on the primitive, then only return to PPO once a scripted diagnostic proves the primitive can produce at least occasional 3+ useful bounces.

## Do Not Spend Time On

Do not continue these unless there is a clear new reason:

- plain `position_strike` PPO with more timesteps,
- reward-weight tuning without a new primitive,
- constant XY residual sweeps,
- tilt-only sweeps,
- simple late z-pulse experiments,
- linear `anchor - intercept` XY gain only,
- using the oracle clamp as a training solution.

Those paths already failed or plateaued around `max_useful_bounces <= 2`.

## Non-Negotiable Gate

Before serious PPO training, a scripted or heuristic policy must pass this gate:

- `max_useful_bounces >= 3` in diagnostic runs,
- `three_or_more_useful_bounce_rate > 0`,
- preferably `three_or_more_useful_bounce_rate >= 0.10` over 100 episodes.

If this gate fails, do not run long PPO. Fix the primitive first.

## First Implementation Target: Contact-Frame Primitive

Add a new action mode rather than mutating existing modes blindly.

Suggested name:

`position_contact_frame`

Suggested action shape:

```text
[radial_offset, tangential_offset, strike_z_residual, pitch_residual, roll_residual]
```

Interpretation:

- `radial_offset`: move the target contact point along the direction that sends the ball back toward the racket/robot hittable region.
- `tangential_offset`: adjust sideways contact point without changing the main return direction too much.
- `strike_z_residual`: small controlled upward/downward adjustment near contact.
- `pitch_residual`, `roll_residual`: small racket normal adjustment.

The key change is that XY action should not be raw world XY. It should be defined in a contact/return frame.

Candidate frame:

```python
return_xy = controller_anchor_xy - predicted_ball_intercept_xy
radial = normalize(return_xy)
tangent = np.array([-radial[1], radial[0]])
contact_offset_xy = radial * radial_offset + tangent * tangential_offset
target_xy = predicted_ball_intercept_xy + contact_offset_xy
```

Fallback when `return_xy` is too small:

- use projected incoming ball velocity direction,
- or racket local axes,
- or projected contact normal from the most recent contact trace.

Keep action scales small at first. This is a contact geometry primitive, not a wide exploration primitive.

## Implementation Surfaces

Expected files to modify:

- `pingpong_rl2/src/pingpong_rl2/envs/keepup_env.py`
- `pingpong_rl2/src/pingpong_rl2/controllers/heuristic_keepup.py`
- `pingpong_rl2/scripts/run_heuristic_keepup_diagnostic.py`

Possibly modify:

- `pingpong_rl2/src/pingpong_rl2/envs/pingpong_sim.py` only if more contact trace fields are needed.
- `pingpong_rl2/scripts/run_ppo_learning.py` only after the scripted gate passes.

Do not start by editing `run_ppo_learning.py`. PPO preset comes later.

## Diagnostic Plan

Run in this order:

1. Small smoke diagnostic, 2 to 5 episodes, to check no crash.
2. 30 episode diagnostic, to see whether any episode reaches 3 useful bounces.
3. 100 episode diagnostic only if 30 episode run has at least one promising 3+ result.
4. Viewer inspection only for the best diagnostic variant.

Required report metrics:

- `mean_useful_bounces`
- `max_useful_bounces`
- `one_or_more_useful_bounce_rate`
- `two_or_more_useful_bounce_rate`
- `three_or_more_useful_bounce_rate`
- top failure reasons
- representative contact trace for best and failed episodes

## If Contact-Frame Primitive Fails

Do not go back to reward tuning yet.

Next primitive to try:

`position_contact_frame_timing`

Suggested action shape:

```text
[radial_offset, tangential_offset, contact_lead_time, z_velocity_bias, normal_pitch]
```

Purpose:

Let the policy shift the strike window earlier/later and add upward racket velocity at impact, instead of only changing a static target pose.

This is different from a simple z pulse. The pulse must be tied to predicted contact time or closest approach time.

## Reward Direction After Primitive Exists

Once scripted gate passes, simplify reward around the real objective:

- reward useful upward bounce,
- reward next predicted ball apex/landing being inside hittable region,
- penalize sending the ball away from the robot,
- penalize floor/out-of-bounds,
- keep smoothness/control penalties secondary.

Avoid many overlapping shaping terms that can cancel each other.

## PPO Only After Gate

Only after scripted `max_useful_bounces >= 3`:

- add a PPO preset for the new action mode,
- start with short training such as 100k or 200k timesteps,
- compare against the scripted primitive and current best baseline,
- do not spend 1M timesteps until the short run shows clear movement in the right metric.

## Final Deliverable

Create a report:

`pingpong_rl2/docs/report/21_contact_frame_primitive_report.md`

The report must include:

- what primitive was added,
- exact commands run,
- metric table,
- whether the scripted gate passed,
- whether PPO is justified yet,
- next single action if it still fails.

## Decision Rule

If the new primitive cannot make the scripted policy produce occasional 3+ useful bounces, the project is not ready for PPO. The next step is still action/control design, not training duration or reward-weight tuning.

