import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import math

# =====================
# 환경 상수
# =====================
MAP_SIZE = 100
DT = 0.1

MAX_CAPACITY = 50.0
BATTERY_MAX = 100.0

BATTERY_MOVE_COST = 0.4
BATTERY_COLLECT_COST = 0.3
BATTERY_IDENTIFY_COST = 0.2

CURRENT_CHANGE_INTERVAL = 40
SONAR_RANGE = 10
IDENTIFY_TIME = 1.0   # ✅ CV 판별 시간 1초

# =====================
# 해양 객체
# =====================
class OceanObject:
    def __init__(self, obj_type):
        self.type = obj_type
        self.detected = obj_type != "seabed_trash"

        self.x = random.uniform(10, MAP_SIZE - 10)
        self.y = random.uniform(10, MAP_SIZE - 10)

        self.visual_size = random.uniform(2.5, 5.0)

        if obj_type == "surface_trash":
            self.real_amount = random.uniform(5, 12)
            self.move_factor = 1.0

        elif obj_type == "animal":
            self.real_amount = 0.0
            self.move_factor = 1.0

        else:  # seabed_trash
            self.real_amount = random.uniform(10, 25)
            self.move_factor = 0.25
            self.y -= random.uniform(5, 10)

        self.vx = random.uniform(-0.3, 0.3)
        self.vy = random.uniform(-0.3, 0.3)

    def move(self, current, locked=False):
        if locked:
            return

        self.x += (self.vx + current[0]) * self.move_factor
        self.y += (self.vy + current[1]) * self.move_factor

        if self.x <= 0 or self.x >= MAP_SIZE:
            self.vx *= -1
        if self.y <= 0 or self.y >= MAP_SIZE:
            self.vy *= -1

        self.x = np.clip(self.x, 0, MAP_SIZE)
        self.y = np.clip(self.y, 0, MAP_SIZE)

# =====================
# 드론
# =====================
class Drone:
    def __init__(self):
        self.x, self.y = BASE
        self.speed = 12
        self.load = 0.0
        self.battery = BATTERY_MAX
        self.state = "IDLE"
        self.target = None
        self.timer = 0

    def dist(self, obj):
        return math.hypot(self.x - obj.x, self.y - obj.y)

    def move_to(self, x, y):
        dx, dy = x - self.x, y - self.y
        d = math.hypot(dx, dy)
        if d > 0:
            self.x += (dx / d) * self.speed * DT
            self.y += (dy / d) * self.speed * DT
            self.battery -= BATTERY_MOVE_COST

# =====================
# 우선순위
# =====================
def priority(drone, obj):
    if obj.type not in ["surface_trash", "seabed_trash"]:
        return -999
    return obj.real_amount * 10 - drone.dist(obj)

def select_target(drone, objs):
    candidates = [
        o for o in objs
        if o.detected and o.real_amount > 0
        and drone.load < MAX_CAPACITY
    ]
    return max(candidates, key=lambda o: priority(drone, o)) if candidates else None

# =====================
# 초기화
# =====================
BASE = (50, 50)

objects = (
    [OceanObject("surface_trash") for _ in range(7)] +
    [OceanObject("animal") for _ in range(4)] +
    [OceanObject("seabed_trash") for _ in range(6)]
)

drone = Drone()
current = np.array([random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4)])
frame_count = 0

# =====================
# 시각화
# =====================
fig, ax = plt.subplots(figsize=(8, 8))

def update(frame):
    global current, frame_count
    frame_count += 1

    ax.clear()
    ax.set_xlim(0, MAP_SIZE)
    ax.set_ylim(0, MAP_SIZE)
    ax.set_title("CV + Sonar 기반 해양 쓰레기 수거 시뮬레이션")

    if frame_count % CURRENT_CHANGE_INTERVAL == 0:
        current = np.array([
            random.uniform(-0.5, 0.5),
            random.uniform(-0.5, 0.5)
        ])

    for obj in objects:
        locked = (drone.state in ["COLLECTING", "IDENTIFYING"] and drone.target == obj)
        obj.move(current, locked)

        if obj.type == "seabed_trash" and not obj.detected:
            if drone.dist(obj) < SONAR_RANGE:
                obj.detected = True

    # =====================
    # 드론 FSM
    # =====================
    if drone.state == "IDLE":
        drone.target = select_target(drone, objects)
        if drone.target:
            drone.state = "MOVING"

    elif drone.state == "MOVING":
        drone.move_to(drone.target.x, drone.target.y)
        if drone.dist(drone.target) < 1.2:
            drone.timer = IDENTIFY_TIME
            drone.state = "IDENTIFYING"

    elif drone.state == "IDENTIFYING":
        drone.timer -= DT
        drone.battery -= BATTERY_IDENTIFY_COST

        if drone.timer <= 0:
            if drone.target.type == "animal":
                objects.remove(drone.target)
                drone.target = None
                drone.state = "IDLE"
            else:
                drone.timer = drone.target.real_amount * 0.3
                drone.state = "COLLECTING"

    elif drone.state == "COLLECTING":
        drone.timer -= DT
        drone.battery -= BATTERY_COLLECT_COST

        if drone.timer <= 0:
            collectable = min(
                drone.target.real_amount,
                MAX_CAPACITY - drone.load
            )
            drone.load += collectable
            drone.target.real_amount -= collectable

            if drone.target.real_amount <= 0:
                objects.remove(drone.target)

            drone.target = None
            drone.state = "IDLE"

    if drone.battery <= 20 or drone.load >= MAX_CAPACITY:
        drone.state = "RETURN"
        drone.target = None

    if drone.state == "RETURN":
        drone.move_to(BASE[0], BASE[1])
        if math.hypot(drone.x - BASE[0], drone.y - BASE[1]) < 1.5:
            drone.load = 0
            drone.battery = BATTERY_MAX
            drone.state = "IDLE"

    # =====================
    # 렌더링
    # =====================
    for obj in objects:
        if not obj.detected:
            continue

        if obj.type == "animal":
            color = "orange"
            label = f"Animal ({obj.visual_size:.1f})"
        elif obj.type == "seabed_trash":
            color = "purple"
            label = f"Seabed {obj.real_amount:.1f}"
        else:
            color = "green"
            label = f"Trash {obj.real_amount:.1f}"

        ax.scatter(obj.x, obj.y, s=obj.visual_size * 35, c=color)
        ax.text(obj.x + 0.8, obj.y + 0.8, label, fontsize=7)

    ax.scatter(drone.x, drone.y, c="red", s=120)
    ax.scatter(BASE[0], BASE[1], c="blue", s=200, marker="s")

    ax.add_patch(
        plt.Circle((drone.x, drone.y), SONAR_RANGE,
                   color="cyan", fill=False, linestyle="dotted")
    )
    ax.quiver(50, 50, current[0] * 20, current[1] * 20, color="cyan")

    ax.text(1, 96, f"State: {drone.state}")
    ax.text(1, 93, f"Load: {drone.load:.1f}/{MAX_CAPACITY}")
    ax.text(1, 90, f"Battery: {drone.battery:.1f}")

ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
plt.show()
