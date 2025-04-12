extends Node3D

@export var ray_distance: float = 1000
@onready var ray: RayCast3D = $RayCast3D
@onready var target: RigidBody3D = $Target

func _ready():
	ray.enabled = true

func _physics_process(delta):
	if !is_instance_valid(target):
		return
	var direction = (target.global_transform.origin - global_transform.origin).normalized()
	ray.target_position = direction * ray_distance
	ray.force_raycast_update()

	if ray.is_colliding():
		var hit = ray.get_collider()

		if hit == target:
			print("✅ ยิงโดน Target!")
		else:
			print("❌ มีสิ่งกีดขวาง: ", hit.name)
	else:
		print("⚠️ ไม่ชนอะไรเลย")
