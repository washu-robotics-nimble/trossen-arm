"""
Write arbitrary text on the calibrated whiteboard.

Requires a calibration from control/scripts/calibrate_whiteboard.py.
The complete trajectory is planned (all IK solved and bounds-checked) before
the arm moves; planning failures abort with no motion.

Usage:
  # preview only — plans everything and saves a PNG, robot not needed
  python control/scripts/write_text.py "Hello World!" --dry-run

  # write it (marker already gripped from calibration; else it prompts)
  python control/scripts/write_text.py "Hello World!"

  # smaller text at a specific board position (meters from bottom-left)
  python control/scripts/write_text.py "Hi" --height 0.04 --start-u 0.05 --start-v 0.20
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from control.writing.contact import probe_contact, diagnostic_descent, ContactError
from control.writing.glyphs import text_to_strokes_ttf, default_cjk_font
from control.writing.hershey import text_to_strokes, preview_strokes
from control.writing.jog import jog_to_point
from control.writing.kinematics import forward_kinematics, inverse_kinematics
from control.writing.safety import home_on_interrupt, glide_home
from control.writing.trajectory import plan_strokes, execute, PlanningError
from control.writing.whiteboard import WhiteboardPlane

HOME = np.zeros(6)
# forward-reaching pose used across existing scripts (see draw_a.py)
READY = np.array([0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Text to write; use \\n for new lines")
    parser.add_argument("--height", type=float, default=0.06, help="Capital letter height (m)")
    parser.add_argument("--fit", action="store_true",
                        help="Auto-scale letter height so the text fits the board")
    parser.add_argument("--margin", type=float, default=0.01,
                        help="Keep text this far (m) from the board edges")
    parser.add_argument("--start-u", type=float, default=None,
                        help="Left edge of text on the board (m); default: centered")
    parser.add_argument("--start-v", type=float, default=None,
                        help="Bottom edge of text on the board (m); default: centered")
    parser.add_argument("--font", default="futural",
                        help="Hershey font for Latin text (futural, futuram, cursive, ...)")
    parser.add_argument("--font-file", default=None,
                        help="Path to a .ttf/.otf/.ttc font for non-Latin text (e.g. "
                             "Chinese). Auto-used for non-ASCII text; defaults to a "
                             "system CJK font. TTF glyphs are drawn as outlines.")
    parser.add_argument("--speed", type=float, default=0.04, help="Writing speed (m/s)")
    parser.add_argument("--hover", type=float, default=0.03, help="Pen-up hover distance (m)")
    parser.add_argument("--press", type=float, default=0.002,
                        help="Press depth past the board plane (m)")
    parser.add_argument("--probe", action="store_true",
                        help="Before writing, feel for the real board surface (contact "
                             "probe) and write relative to it — compensates marker shift")
    parser.add_argument("--probe-tune", action="store_true",
                        help="Descend and print effort readings WITHOUT writing, to tune "
                             "--contact-threshold. Does not draw anything.")
    parser.add_argument("--touch-off", action="store_true",
                        help="Manually jog the marker until it just touches the board, "
                             "press SPACE to set the writing depth (no force sensing). "
                             "The reliable way to compensate marker shift on this arm.")
    parser.add_argument("--contact-threshold", type=float, default=2.0,
                        help="Signed joint-1 effort rise (Nm) counted as board contact; "
                             "~2.0 validated on the real arm (probe prints readings)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan and preview only; do not connect to the robot")
    parser.add_argument("--preview", default="text_preview.png", help="Preview PNG path")
    parser.add_argument("--skip-gripper", action="store_true",
                        help="Skip the marker-gripping prompt (marker already held)")
    parser.add_argument("--calibration", default=None,
                        help="Path to a whiteboard calibration JSON (default: config/)")
    args = parser.parse_args()

    text = args.text.replace("\\n", "\n")

    plane = (WhiteboardPlane.load(args.calibration) if args.calibration
             else WhiteboardPlane.load())
    print(f"Board: {plane.width*100:.1f} x {plane.height*100:.1f} cm "
          f"(tip length {plane.tip_length} m)")

    # Non-Latin text (e.g. Chinese) can't use the Hershey single-stroke fonts;
    # fall back to outline glyphs from a TTF font.
    use_ttf = bool(args.font_file) or any(ord(c) > 0x7f for c in text)

    def render(char_height):
        if use_ttf:
            return text_to_strokes_ttf(text, char_height=char_height,
                                       font_path=args.font_file)
        return text_to_strokes(text, char_height=char_height, font=args.font)

    if use_ttf:
        fpath = args.font_file or default_cjk_font()
        print(f"Non-Latin text -> outline glyphs from {fpath}")

    strokes, w, h = render(args.height)
    if not strokes:
        print("Nothing to write.")
        return

    # Usable area after margins.
    usable_w = plane.width - 2 * args.margin
    usable_h = plane.height - 2 * args.margin

    # Largest letter height that still fits both dimensions (text size scales
    # linearly with char height).  The 0.97 keeps the fitted text a hair
    # inside the margin instead of exactly on the boundary.
    fit_scale = min(usable_w / w, usable_h / h) if w > 0 and h > 0 else 1.0
    fit_scale *= 0.97
    max_height = args.height * fit_scale

    if args.fit and fit_scale < 1.0:
        print(f"Auto-fit: shrinking letters {args.height*100:.1f} -> "
              f"{max_height*100:.1f} cm to fit the board.")
        strokes, w, h = render(max_height)
    elif not args.fit and fit_scale < 1.0:
        print(f"\nText block ({w*100:.1f} x {h*100:.1f} cm) is too big for the "
              f"{plane.width*100:.1f} x {plane.height*100:.1f} cm board.")
        print(f"  -> use --fit to auto-scale, or --height {max_height:.3f} "
              f"(<= {max_height*100:.1f} cm) to fit manually.")
        sys.exit(1)

    print(f"Text block: {w*100:.1f} x {h*100:.1f} cm, {len(strokes)} strokes")

    u0 = args.start_u if args.start_u is not None else (plane.width - w) / 2
    v0 = args.start_v if args.start_v is not None else (plane.height - h) / 2
    strokes = [s + np.array([u0, v0]) for s in strokes]

    def plan(surface_offset=0.0):
        try:
            return plan_strokes(
                plane, strokes, READY,
                write_speed=args.speed, hover=args.hover, press=args.press,
                margin=args.margin, surface_offset=surface_offset,
            )
        except PlanningError as e:
            print(f"\nPLANNING FAILED — no motion sent.\n{e}")
            sys.exit(1)

    print("Planning trajectory...")
    waypoints = plan()
    total_t = sum(wp.duration for wp in waypoints)
    print(f"Planned {len(waypoints)} waypoints, estimated {total_t:.0f} s")

    preview_strokes(strokes, args.preview)
    print(f"Preview saved to {args.preview}")

    if args.dry_run:
        print("Dry run — done.")
        return

    # board coordinate at the text center — where we probe for contact
    text_center_uv = (u0 + w / 2, v0 + h / 2)

    import trossen_arm
    import yaml

    with open(os.path.join(os.path.dirname(__file__), "../../config/robot_config.yaml")) as f:
        cfg = yaml.safe_load(f)

    print("\nConnecting to the robot...")
    driver = trossen_arm.TrossenArmDriver()

    # configure()'s 4th arg is clear_error — clear any stale fault on startup.
    def _configure(clear_error):
        driver.configure(
            trossen_arm.Model.wxai_v0,
            trossen_arm.StandardEndEffector.wxai_v0_leader,
            cfg["robot"]["ip"],
            clear_error,
        )

    _configure(True)

    # From here on, any Ctrl-C or crash lifts the marker and glides home
    # instead of leaving it jammed against the board.  clear_error lets a
    # faulted arm be recovered before homing.
    with home_on_interrupt(driver, clear_error=lambda: _configure(True)):
        if not args.skip_gripper:
            print("Opening the gripper...")
            driver.set_gripper_mode(trossen_arm.Mode.external_effort)
            driver.set_gripper_external_effort(20.0, 3.0, True)
            input("\nPlace the marker in the gripper (same as during calibration).\n"
                  "Press Enter to close the gripper...")
            driver.set_gripper_external_effort(-20.0, 3.0, True)
            time.sleep(0.5)

        input(f"\nAbout to write on the board ({total_t:.0f} s). "
              "Clear the area and press Enter to start...")

        driver.set_arm_modes(trossen_arm.Mode.position)
        print("Moving to ready pose...")
        driver.set_arm_positions(READY, 3.0, True)

        if args.probe_tune:
            print("\nPROBE TUNING — descending and printing readings, NOT writing.")
            print("Watch for where 'effort per joint' jumps sharply — that's contact.")
            _, R_ready = forward_kinematics(READY, plane.tip_length)
            diagnostic_descent(driver, plane, text_center_uv, R_ready[:, 0],
                               hover=args.hover)
            print("\nTuning descent done. Pick --contact-threshold above the "
                  "free-air rise but below the contact jump.")
            glide_home(driver, clear_error=lambda: _configure(True))
            return

        if args.touch_off:
            print("\nMANUAL TOUCH-OFF: jog the marker until it JUST touches the board.")
            _, R_ready = forward_kinematics(READY, plane.tip_length)
            target_dir = R_ready[:, 0]
            # pre-position at a hover over the text center, then hand off to jog
            hover_pos = plane.point(*text_center_uv, offset=args.hover)
            q_hover, ok, _ = inverse_kinematics(
                hover_pos, target_dir=target_dir, q_init=READY,
                tip_length=plane.tip_length)
            if ok:
                driver.set_arm_positions(q_hover, 2.0, True)
            else:
                q_hover = READY
            q_touch, pos_touch = jog_to_point(
                driver, q_hover, plane.tip_length, target_dir,
                "SURFACE ('w'=toward board, 's'=back off; SPACE when it just touches)")
            if q_touch is None:
                print("\nTouch-off aborted — homing.")
                glide_home(driver, clear_error=lambda: _configure(True))
                return
            surface_offset = float((pos_touch - plane.origin) @ plane.normal)
            print(f"Writing surface set to {surface_offset*1000:+.1f} mm from "
                  "calibration.")
            waypoints = plan(surface_offset=surface_offset)

        elif args.probe:
            print("\nProbing for the real board surface (feeling for contact)...")
            _, R_ready = forward_kinematics(READY, plane.tip_length)
            try:
                surface_offset = probe_contact(
                    driver, plane, text_center_uv, R_ready[:, 0],
                    threshold=args.contact_threshold, hover=args.hover,
                )
            except ContactError as e:
                print(f"\nCONTACT PROBE FAILED — not writing.\n{e}")
                glide_home(driver, clear_error=lambda: _configure(True))
                return
            print(f"Surface found {surface_offset*1000:+.1f} mm from calibration; "
                  "re-planning to track it.")
            waypoints = plan(surface_offset=surface_offset)

        print("Writing...")
        execute(driver, waypoints)

        print("Returning to ready pose...")
        driver.set_arm_positions(READY, 3.0, True)

        input("\nDone. Press Enter to return the arm to home...")
        glide_home(driver, clear_error=lambda: _configure(True))
        print("Finished.")


if __name__ == "__main__":
    main()
