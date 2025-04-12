
# ความเร็วในการเคลื่อนที่
var speed = 5.0
var velocity = Vector3.ZERO

func _process(delta):
	velocity = Vector3.ZERO  # รีเซ็ตความเร็ว

	# รับข้อมูลจากปุ่ม WASD หรือ Arrow keys
	if Input.is_action_pressed("move_up"):
		velocity.z -= speed
	if Input.is_action_pressed("move_down"):
		velocity.z += speed
	if Input.is_action_pressed("move_left"):
		velocity.x -= speed
	if Input.is_action_pressed("move_right"):
		velocity.x += speed

	# ทำให้ตัวละครเคลื่อนที่
	move_and_slide()
