from physica import *

ball = Object2(
    "ball",
    mass=0.45,
    position=Vector2(0, 10),
    velocity=Vector2(5, 0)
)

@continuous(ball)
class Gravity:
    gravity = Vector2(0, -4.4145)

@update(ball)
def physics(obj, dt):
    obj.integrate(dt)

for _ in range(100):
    physics(ball, 0.016)

print(ball.position)
print(ball.speed())
print(ball.kinetic_energy())