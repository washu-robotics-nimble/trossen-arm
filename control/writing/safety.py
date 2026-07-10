"""
Shared arm-safety helpers: glide the arm home, and a context manager that
does so automatically when a script is interrupted or terminated.

IMPORTANT — what this does and does NOT guarantee:
The WXAI holds its position with onboard servos in both position and idle
mode ("idle : All joints are braked"), so the arm does NOT fall just because
this Python process dies.  This helper repositions the arm to a safe home on
a *graceful* stop.  It CANNOT run on SIGKILL (kill -9), on an interpreter
crash, or on power loss / E-stop — and on power loss the arm has no
mechanical brakes, so it will sag.  Only a physical cradle and stable power
protect against those.

Covered stop signals: Ctrl-C (SIGINT), uncaught exceptions, SIGTERM (kill),
and SIGHUP (terminal closed).

A subtle failure mode this handles: if the arm is in an ERROR state, the
driver silently ignores position commands, so a naive home command does
nothing.  glide_home() detects that, clears the error if it can, and verifies
the arm actually reached home — warning loudly if it did not.
"""

import signal
from contextlib import contextmanager

import numpy as np
import trossen_arm

HOME = np.zeros(6)

# how close (rad, per joint) counts as "arrived at home"
_ARRIVAL_TOL = 0.08


class _Terminated(Exception):
    """Raised inside the guarded block when SIGTERM/SIGHUP is received."""

    def __init__(self, signame):
        self.signame = signame
        super().__init__(signame)


def glide_home(driver, speed: float = 0.8, min_time: float = 2.0,
               clear_error=None) -> float:
    """Move all joints to zero, scaling duration to the distance travelled.

    Args:
        driver: configured TrossenArmDriver.
        speed: approximate joint speed (rad/s).
        min_time: minimum move duration (s).
        clear_error: optional callable that clears the arm's error state
            (e.g. re-configuring with clear_error=True).  Invoked only if an
            error is detected, since an errored arm ignores motion commands.

    Returns:
        The move duration used.  Prints a loud warning if the arm does not
        reach home (e.g. because of an unclearable fault).
    """
    # An errored arm silently ignores position commands — detect and clear.
    # get_error_information() returns the literal "No error" when healthy.
    try:
        err = (driver.get_error_information() or "").strip()
    except Exception:
        err = ""
    if err and err.lower() != "no error":
        print(f"  ARM ERROR STATE: {err}")
        if clear_error is not None:
            print("  Clearing error before homing...")
            try:
                clear_error()
            except Exception as e:
                print(f"  Could not clear error automatically: {e}")

    # get_positions() returns 6 arm joints + gripper; take the arm joints.
    current = np.array(driver.get_positions()[:6], dtype=np.float64)
    max_delta = float(np.max(np.abs(current - HOME)))
    duration = max(max_delta / speed, min_time)

    driver.set_arm_modes(trossen_arm.Mode.position)
    driver.set_arm_positions(HOME, duration, True)

    # Verify the arm actually arrived — a rejected command leaves it put.
    final = np.array(driver.get_positions()[:6], dtype=np.float64)
    residual = float(np.max(np.abs(final - HOME)))
    if residual > _ARRIVAL_TOL:
        print(f"  !! WARNING: arm did NOT reach home (off by "
              f"{np.degrees(residual):.0f} deg on at least one joint).")
        print(f"     Current pose: {np.round(final, 3)}")
        print("     The arm may be in a fault state. HOLD/SUPPORT IT, then "
              "power-cycle the controller and run home_arm.py.")
    return duration


def _try_home(driver, speed, min_time, clear_error):
    print("\n\nStopping — gliding the arm home safely. DO NOT TOUCH THE ARM.")
    try:
        glide_home(driver, speed, min_time, clear_error)
        print("Arm homing command complete.")
    except KeyboardInterrupt:
        print("\n!! Homing cancelled. HOLD THE ARM — it may be unsupported.")
    except Exception as home_err:
        print(f"\n!! Could not home the arm automatically: {home_err}")
        print("   HOLD THE ARM — it may be unsupported.")


@contextmanager
def home_on_interrupt(driver, speed: float = 0.8, min_time: float = 2.0,
                      clear_error=None):
    """Glide the arm home if the wrapped block is interrupted or terminated.

    Handles Ctrl-C (SIGINT), SIGTERM (kill), SIGHUP (terminal closed), and any
    uncaught exception: the arm is homed, then the process exits (130 for
    signals) or the original exception is re-raised so the error stays visible.
    Normal completion does nothing (the script handles its own end-of-run
    homing).

    Pass clear_error (a callable) so a faulted arm can be recovered before
    homing — otherwise an errored arm will ignore the home command.

    CANNOT protect against SIGKILL, interpreter crashes, or power loss.
    """
    installed = {}

    def _raise_terminated(signum, frame):
        raise _Terminated(signal.Signals(signum).name)

    for name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            installed[sig] = signal.signal(sig, _raise_terminated)
        except (ValueError, OSError):
            pass  # not on main thread, or unsupported platform

    try:
        yield
    except KeyboardInterrupt:
        _try_home(driver, speed, min_time, clear_error)
        raise SystemExit(130)
    except _Terminated as term:
        print(f"\n[received {term.signame}]", end="")
        _try_home(driver, speed, min_time, clear_error)
        raise SystemExit(130)
    except Exception:
        _try_home(driver, speed, min_time, clear_error)
        raise
    finally:
        for sig, prev in installed.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass
