import math
import time
import os
from typing import List, Tuple, Dict

try:
    from interbotix_xs_modules.arm import InterbotixManipulatorXS
except Exception:
    InterbotixManipulatorXS = None

Point2D = Tuple[float, float]
Stroke = List[Point2D]

def _polyline(points: List[Point2D]) -> Stroke:
    return points

def _rect(x0, y0, x1, y1) -> Stroke:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]

def vector_font_AZ(h: float) -> Dict[str, List[Stroke]]:
    w = 0.6 * h
    m = w / 2.0
    t = 0.15 * h
    font = {}
    font['A'] = [_polyline([(0,0),(0,h)]), _polyline([(w,0),(w,h)]), _polyline([(0,h*0.6),(w,h*0.6)])]
    font['B'] = [_polyline([(0,0),(0,h)]), _polyline([(0,h),(m,h),(w,h*0.8),(m,h*0.6),(0,h*0.6)]), _polyline([(0,h*0.6),(m,h*0.6),(w,h*0.4),(m,h*0.2),(0,0)])]
    font['C'] = [_polyline([(w,0),(0,0),(0,h),(w,h)])]
    font['D'] = [_polyline([(0,0),(0,h)]), _polyline([(0,h),(m,h),(w,h*0.75),(w,h*0.25),(m,0),(0,0)])]
    font['E'] = [_polyline([(w,0),(0,0),(0,h),(w,h)]), _polyline([(0,h*0.5),(m,h*0.5)])]
    font['F'] = [_polyline([(0,0),(0,h),(w,h)]), _polyline([(0,h*0.5),(m,h*0.5)])]
    font['G'] = [_polyline([(w,0),(0,0),(0,h),(w,h),(w,h*0.5),(m,h*0.5)])]
    font['H'] = [_polyline([(0,0),(0,h)]), _polyline([(w,0),(w,h)]), _polyline([(0,h*0.5),(w,h*0.5)])]
    font['I'] = [_polyline([(0,h),(w,h)]), _polyline([(m,h),(m,0)]), _polyline([(0,0),(w,0)])]
    font['J'] = [_polyline([(0,h),(w,h)]), _polyline([(m,h),(m,0),(0,0)])]
    font['K'] = [_polyline([(0,0),(0,h)]), _polyline([(0,h*0.5),(w,h)]), _polyline([(0,h*0.5),(w,0)])]
    font['L'] = [_polyline([(0,h),(0,0),(w,0)])]
    font['M'] = [_polyline([(0,0),(0,h)]), _polyline([(w,0),(w,h)]), _polyline([(0,h),(m,0),(w,h)])]
    font['N'] = [_polyline([(0,0),(0,h)]), _polyline([(w,0),(w,h)]), _polyline([(0,h),(w,0)])]
    font['O'] = [ _rect(0,0,w,h) ]
    font['P'] = [_polyline([(0,0),(0,h),(m,h),(w,h*0.75),(m,h*0.5),(0,h*0.5)])]
    font['Q'] = [ _rect(0,0,w,h), _polyline([(m,h*0.4),(w,0)]) ]
    font['R'] = [_polyline([(0,0),(0,h),(m,h),(w,h*0.75),(m,h*0.5),(0,h*0.5)]), _polyline([(0,h*0.5),(w,0)])]
    font['S'] = [_polyline([(w,0),(0,0),(0,h*0.5),(w,h*0.5),(w,h),(0,h)])]
    font['T'] = [_polyline([(0,h),(w,h)]), _polyline([(m,h),(m,0)])]
    font['U'] = [_polyline([(0,h),(0,t),(w,t),(w,h)])]
    font['V'] = [_polyline([(0,h),(m,0),(w,h)])]
    font['W'] = [_polyline([(0,h),(m,0),(w,h),(w*0.5,h*0.4),(0,h)])]
    font['X'] = [_polyline([(0,0),(w,h)]), _polyline([(0,h),(w,0)])]
    font['Y'] = [_polyline([(0,h),(m,h*0.5)]), _polyline([(w,h),(m,h*0.5),(m,0)])]
    font['Z'] = [_polyline([(0,h),(w,h),(0,0),(w,0)])]
    return font

def strokes_for_letter(ch: str, h: float) -> List[Stroke]:
    font = vector_font_AZ(h)
    c = ch.upper()
    if c in font:
        return font[c]
    return [ _rect(0,0,0.6*h,h) ]

def text_to_strokes(text: str, char_h: float = 0.03, spacing: float = 0.015) -> List[Stroke]:
    x_cursor = 0.0
    strokes: List[Stroke] = []
    for ch in text:
        if ch == ' ':
            x_cursor += char_h*0.6 + spacing
            continue
        glyph = strokes_for_letter(ch, char_h)
        for poly in glyph:
            strokes.append([(x + x_cursor, y) for (x, y) in poly])
        x_cursor += char_h*0.6 + spacing
    return strokes

class TrossenArm:
    def __init__(self, model: str = None, group_name: str = "arm", gripper_name: str = "gripper"):
        if InterbotixManipulatorXS is None:
            self.bot = None
        else:
            self.bot = InterbotixManipulatorXS(model or os.getenv("INTERBOTIX_ROBOT_MODEL","wx250s"), group_name, gripper_name)
    def home(self):
        if self.bot: self.bot.arm.go_to_home_pose()
        return True
    def sleep(self):
        if self.bot: self.bot.arm.go_to_sleep_pose()
        return True
    def open_gripper(self):
        if self.bot: self.bot.gripper.open()
        return True
    def close_gripper(self):
        if self.bot: self.bot.gripper.close()
        return True
    def move_ee_abs(self, x=None,y=None,z=None,roll=0.0,pitch=math.radians(90.0),yaw=0.0):
        if self.bot:
            self.bot.arm.set_ee_pose_components(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, moving_time=1.0, accel_time=0.25)
        else:
            time.sleep(0.005)
        return True
    def move_ee_lin(self, dx=0.0,dy=0.0,dz=0.0,roll=0.0,pitch=math.radians(90.0),yaw=0.0,speed=0.04):
        if self.bot:
            self.bot.arm.set_ee_cartesian_trajectory(x=dx, y=dy, z=dz, roll=roll, pitch=pitch, yaw=yaw, speed=speed)
        else:
            time.sleep(0.005)
        return True
    def draw_stroke(self, stroke: Stroke, origin_xyz: Tuple[float,float,float], z_draw: float, z_safe: float, yaw: float = 0.0, speed: float = 0.04):
        if not stroke: return True
        ox, oy, oz = origin_xyz
        sx, sy = stroke[0]
        self.move_ee_abs(ox+sx, oy+sy, oz+z_safe, yaw=yaw)
        self.move_ee_lin(0.0, 0.0, z_draw - z_safe, yaw=yaw, speed=speed)
        px, py = sx, sy
        for (nx, ny) in stroke[1:]:
            self.move_ee_lin((nx-px), (ny-py), 0.0, yaw=yaw, speed=speed)
            px, py = nx, ny
        self.move_ee_lin(0.0, 0.0, z_safe - z_draw, yaw=yaw, speed=speed)
        return True
    def draw_strokes(self, strokes: List[Stroke], origin_xyz: Tuple[float,float,float], z_draw: float, z_safe: float, yaw: float = 0.0, speed: float = 0.04):
        for s in strokes:
            self.draw_stroke(s, origin_xyz, z_draw, z_safe, yaw, speed)
        return True

class SmolVLA:
    def parse(self, instruction: str) -> dict:
        t = instruction.strip()
        if t.lower().startswith("write "):
            if len(t) >= 7 and t[6] in ['"',"'"]:
                q = t[6]
                end = t.rfind(q)
                content = t[7:end] if end > 6 else t[6:]
            else:
                content = t[6:]
            return {"task":"write_text","text":content}
        return {"task":"unknown","raw":t}
    def plan(self, task: dict) -> dict:
        if task.get("task") == "write_text":
            return {"plan":"draw_text","text":task["text"],"char_h":0.03,"spacing":0.015}
        return {"plan":"noop"}

class GrootPlanner:
    def synthesize_strokes(self, plan: dict) -> List[Stroke]:
        if plan.get("plan") == "draw_text":
            return text_to_strokes(plan["text"], plan["char_h"], plan["spacing"])
        return []

class LeRobotPolicy:
    def refine(self, strokes: List[Stroke]) -> List[Stroke]:
        return strokes

def draw_letter(arm: TrossenArm, letter: str, origin_xyz=(0.25,0.0,0.0), char_h=0.04, z_draw=0.005, z_safe=0.06, speed=0.04):
    strokes = text_to_strokes(letter, char_h=char_h, spacing=char_h*0.2)
    arm.home()
    arm.open_gripper()
    arm.move_ee_abs(origin_xyz[0], origin_xyz[1], origin_xyz[2]+z_safe)
    arm.draw_strokes(strokes, origin_xyz, z_draw, z_safe, yaw=0.0, speed=speed)
    arm.sleep()

def draw_text_pipeline(instruction: str, origin_xyz=(0.25,0.0,0.0), z_draw=0.005, z_safe=0.06, speed=0.04):
    arm = TrossenArm()
    smolvla = SmolVLA()
    groot = GrootPlanner()
    lerobot = LeRobotPolicy()
    task = smolvla.parse(instruction)
    plan = smolvla.plan(task)
    raw_strokes = groot.synthesize_strokes(plan)
    refined = lerobot.refine(raw_strokes)
    arm.home()
    arm.open_gripper()
    arm.move_ee_abs(origin_xyz[0], origin_xyz[1], origin_xyz[2]+z_safe)
    arm.draw_strokes(refined, origin_xyz, z_draw, z_safe, yaw=0.0, speed=speed)
    arm.sleep()

def main():
    arm = TrossenArm()
    draw_letter(arm, 'A', origin_xyz=(0.25,0.0,0.0), char_h=0.05, z_draw=0.004, z_safe=0.06, speed=0.035)
    draw_text_pipeline('write "Hello World"', origin_xyz=(0.25,0.0,0.0), z_draw=0.004, z_safe=0.06, speed=0.035)

if __name__ == "__main__":
    main()