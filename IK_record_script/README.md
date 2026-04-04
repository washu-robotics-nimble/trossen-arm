# IK Record Script

Kinesthetic teaching toolkit for the Trossen WidowX AI (wxai_v0) arm. Manually move the arm to desired poses, capture joint-space keyframes, and replay them.

## Scripts

- **IK_record.py** — Record keyframes by physically moving the arm in compliant mode. Capture joint positions + gripper state with keyboard commands, save to `.npy`, then replay.
- **replay.py** — Load a saved `.npy` keyframe file and replay the trajectory on the arm.
- **simple_move.py** — Basic test script for arm connection, gripper open/close, and joint movement.
- **zero.py** — Utility functions to return the arm to zero position and reset the gripper.

## Usage

```bash
# Record keyframes (arm enters compliant mode)
python IK_record.py 

steps: 
- r: task name 
- move arm to desire position. note: don't be dramatic since this will be complete in 5 second. 
- k: record current position 
- g: gripper position change to opposite of current status 
- once all frames are recorded. 
- s: save file 
- q: quit 

# Replay a saved recording
python replay.py
```

## Controls (during recording)

| Key | Action |
|-----|--------|
| `r` | Set task name |
| `k` | Capture current joint positions as keyframe |
| `g` | Toggle gripper open/close |
| `p` | Print all keyframes |
| `s` | Save keyframes to file |
| `clear` | Clear all keyframes |
| `e` | Emergency stop |
| `comp` | Return to compliant mode |
| `q` | Quit |

## Output

Keyframes are saved to `IK_RESULTS/<task_name>.npy` as a dict with keys: `task_name`, `joint_pos` (Nx6 array), `gripper_pos` (N bool array).
