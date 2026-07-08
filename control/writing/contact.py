"""
Contact probing ("touch-off") for the whiteboard.

The marker can sit at an unknown depth in the gripper (it slips), so the
calibrated plane may not match where the marker tip actually contacts the
board.  This descends the marker slowly toward the board along the surface
normal until the arm's external-effort sensing detects contact, and reports
the true surface depth as an offset relative to the calibrated plane.

WHY THIS IS HARD ON THIS ARM
The WXAI's external-effort estimate carries ~1 Nm of pose-dependent model
error even in free air, so you CANNOT detect contact by comparing the
absolute effort against a fixed threshold — the noise floor swamps it.
Instead we detect the *sudden rise* in effort as the marker pushes into the
board, measured over a short sliding window of steps (a numerical derivative).
Slow pose-dependent drift cancels out; only a real, growing contact force
trips it.  Two extra guards prevent the earlier "wrote in the air" failure:

  * a plausibility gate — contact is only accepted once the marker is near
    the calibrated plane (not 30 mm above it), and
  * consecutive confirmation — the rise must persist for 2+ steps, so a
    single motion transient can't trigger it.

The window/threshold still need tuning on real hardware; run
diagnostic_descent() first to see what your arm's readings look like.
"""

import time

import numpy as np

from .kinematics import inverse_kinematics


class ContactError(RuntimeError):
    pass


def _mover(driver, plane, uv, target_dir, step, speed=0.6, min_dur=0.3):
    """Return a function that moves the tip to a given normal offset.

    The move duration is scaled to the actual joint distance travelled, so the
    initial approach from the ready pose glides in gently while the small
    per-step probe moves stay quick.  A fixed short duration here would jerk
    the arm violently on that first large move.
    """
    tip = plane.tip_length
    u, v = uv

    def move_to(offset, seed):
        target = plane.point(u, v, offset=offset)
        q, ok, err = inverse_kinematics(target, target_dir=target_dir,
                                        q_init=seed, tip_length=tip)
        if not ok:
            raise ContactError(
                f"IK failed while probing at offset {offset*1000:.0f} mm "
                f"(residual {err*1000:.1f} mm).")
        seed_arr = np.asarray(seed, dtype=np.float64)
        max_delta = float(np.max(np.abs(q - seed_arr)))
        duration = max(max_delta / speed, min_dur)
        driver.set_arm_positions(q, duration, True)
        return q

    return move_to


def _read_effort(driver, samples=5):
    """Median external effort over the arm joints (robust to spikes)."""
    reads = [np.array(driver.get_external_efforts()[:6], dtype=np.float64)
             for _ in range(samples)]
    return np.median(reads, axis=0)


def probe_contact(driver, plane, uv, target_dir, *,
                  threshold=2.0, window=8, contact_joint=1, detect_max=0.012,
                  hover=0.03, max_push=0.02, step=0.001,
                  settle=0.3, confirm=2, verbose=True):
    """Descend toward the board until contact; return the surface offset (m).

    Detection is by the RISE in external effort over a sliding window of
    `window` steps, not an absolute threshold — this arm's effort estimate is
    too noisy in free air for an absolute comparison.

    Args:
        driver: configured TrossenArmDriver, arm already in position mode.
        plane: calibrated WhiteboardPlane.
        uv: (u, v) board coordinate to probe at (e.g. text center).
        target_dir: marker pointing direction (tool x-axis) to hold.
        threshold: SIGNED effort rise (Nm) over `window` steps, on the
            contact joint, that counts as contact.  Validated ~2.0 Nm on the
            real arm; still confirm with diagnostic_descent.
        window: number of steps over which the rise is measured (mm at
            step=0.001).  Larger windows accumulate the monotonic contact
            signal while bounded noise cancels — 8 works well here.
        contact_joint: index of the joint that bears the marker reaction
            force (1 = shoulder, for a vertical board in front of the arm).
            The other joints are too noisy on this arm; watching the loaded
            joint's signed rise is what makes detection reliable.
        detect_max: contact is only accepted once the offset is at/below this
            (m above the calibrated plane) — rejects phantom contact far off
            the board.
        hover: start this far off the calibrated surface (m).
        max_push: probe from +hover down to -max_push past the plane (m).
        step: probe increment (m).
        settle: pause after each step before reading (s) — let motion damp out.
        confirm: require the rise to persist this many consecutive steps.
        verbose: print each reading (for tuning).

    Returns:
        surface_offset (m) relative to the calibrated plane where contact
        began.  Positive = surface nearer the robot than calibrated.

    Raises:
        ContactError if no contact is detected within max_push.
    """
    move_to = _mover(driver, plane, uv, target_dir, step)

    seed = np.array(driver.get_positions()[:6], dtype=np.float64)
    seed = move_to(hover, seed)
    time.sleep(settle)

    if verbose:
        print(f"  watching joint {contact_joint} signed rise over a "
              f"{window}-step window; contact threshold {threshold} Nm")

    history = []          # effort readings per step
    consec = 0
    n_steps = int(round((hover + max_push) / step))

    for k in range(n_steps + 1):
        offset = hover - k * step
        seed = move_to(offset, seed)
        time.sleep(settle)
        effort = _read_effort(driver)
        history.append(effort)

        # signed rise on the loaded joint: contact pushes it one way and the
        # rise accumulates over the window; free-air noise stays bounded.
        rise = 0.0
        if len(history) > window:
            rise = float(effort[contact_joint] - history[-1 - window][contact_joint])

        gated = offset <= detect_max      # only trust contact near the plane
        if verbose:
            flag = "" if gated else "  (above board — ignoring)"
            print(f"  offset {offset*1000:+5.1f} mm  |  joint-{contact_joint} "
                  f"rise {rise:+.3f} Nm{flag}")

        if gated and rise > threshold:
            consec += 1
            if consec >= confirm:
                # detection lags first contact by ~half the window (the rise
                # must accumulate); compensate so we report where it touched.
                contact_offset = offset + (window // 2) * step
                if verbose:
                    print(f"  CONTACT near offset {contact_offset*1000:+.1f} mm "
                          f"(joint-{contact_joint} rise {rise:+.3f} Nm > "
                          f"{threshold} Nm, confirmed {confirm}x)")
                move_to(hover, seed)   # lift off before returning
                return contact_offset
        else:
            consec = 0

    move_to(hover, seed)
    raise ContactError(
        f"No contact within {max_push*1000:.0f} mm past the calibrated plane. "
        f"The marker may be retracted more than that, the threshold ({threshold} "
        f"Nm) may be too high for the effort rise, or the board moved. Run "
        f"diagnostic_descent() to inspect readings, re-seat the marker, or "
        f"recalibrate.")


def diagnostic_descent(driver, plane, uv, target_dir, *,
                       hover=0.03, max_push=0.02, step=0.001, settle=0.3):
    """Descend and print effort readings WITHOUT deciding contact or writing.

    Use this to tune `threshold`/`window`: watch where the effort rise clearly
    jumps as the marker touches the board.  Returns the list of
    (offset, effort_vector) so you can inspect/plot it.  Retreats to hover at
    the end.
    """
    move_to = _mover(driver, plane, uv, target_dir, step)
    seed = np.array(driver.get_positions()[:6], dtype=np.float64)
    seed = move_to(hover, seed)
    time.sleep(settle)

    readings = []
    n_steps = int(round((hover + max_push) / step))
    print("  offset(mm)  effort per joint (Nm)")
    for k in range(n_steps + 1):
        offset = hover - k * step
        seed = move_to(offset, seed)
        time.sleep(settle)
        effort = _read_effort(driver)
        readings.append((offset, effort))
        print(f"  {offset*1000:+6.1f}   {np.round(effort, 3)}")
    move_to(hover, seed)
    return readings
