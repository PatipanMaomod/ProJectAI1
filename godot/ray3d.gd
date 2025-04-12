extends Node3D

@export var ray_distance: float = 1.5
@export var ray_step_deg: float = 10.0
@export var target: Node3D  # drag target มาจาก Inspector
@onready var ray_cast_3d: RayCast3D = $RayCast3D

var has_collision: bool = false
var has_target_collision: bool = false


var target_ray: RayCast3D
var rays := []

func _ready():
	generate_360_rays()

	target_ray = RayCast3D.new()
	target_ray.name = "RayCast3D_Target"
	target_ray.enabled = true
	add_child(target_ray)

func generate_360_rays():
	var ray_count = int(360 / ray_step_deg)
	for i in ray_count:
		var ray = RayCast3D.new()
		var angle_rad = deg_to_rad(i * ray_step_deg)
		var direction = Vector3.FORWARD.rotated(Vector3.UP, angle_rad)
		ray.target_position = direction * ray_distance
		ray.enabled = true
		add_child(ray)
		rays.append(ray)

func _physics_process(delta):
	has_collision = false
	has_target_collision = false

	# ยิงรอบตัว
	for ray in rays:
		if ray.is_colliding():
			has_collision = true
			break

	# ยิง ray พิเศษไปหา target
	if target and target_ray:
		var direction = (target.global_transform.origin - global_transform.origin).normalized()
		target_ray.target_position = direction * ray_distance
		target_ray.force_raycast_update()

		if target_ray.is_colliding():
			var hit = target_ray.get_collider()
			if hit == target:
				has_target_collision = true
				print("🎯 เจอเป้าหมายโดยตรง")

	
