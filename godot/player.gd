extends CharacterBody3D
@onready var target: RigidBody3D = $"../Target"
@onready var wall_2: StaticBody3D = $"../wall/wall2"
@onready var wall_3: StaticBody3D = $"../wall/wall3"
@onready var wall_1: StaticBody3D = $"../wall/wall1"
@onready var ray3d: Node3D = $head




const SPEED = 10
const JUMP_VELOCITY = 4.5
var socket = WebSocketPeer.new()
var has_sent = false
var epoch_count = 0



func _ready():
	var url = "ws://localhost:8765"
	var error = socket.connect_to_url(url)
	
	
	if error == OK:
		print("Connected to server!")
	else:
		print("Failed to connect: ", error)

func _process(delta):
	socket.poll()  # อัปเดตสถานะของ socket
	var state = socket.get_ready_state()
	if socket.get_ready_state() != WebSocketPeer.STATE_OPEN:
		return
	
	if state == WebSocketPeer.STATE_OPEN:
		if not has_sent:
			print("Connection สำเร็จ")
			has_sent = true
			
	elif state == WebSocketPeer.STATE_CLOSED:
		print("Connection closed.")
		set_process(false)
		
func _reset_position():
	position = Vector3(-13,1,0)

func _physics_process(delta: float) -> void:
	if socket.get_available_packet_count() > 0:
		var message = socket.get_packet().get_string_from_utf8()
		if not message:
			socket.send_text("ping")
		if message:
			var saw_wall = ray3d.has_collision
			var input = Vector2(0, 0)
			if message == "forward":
				input.x = 1
			elif message == "backward":
				input.x = -1
			elif message == "left":
				input.y = -1
			elif message == "right":
				input.y = 1
			elif message == "stop":
				input.y = 0
				input.x = 0
				
			var direction := (transform.basis * Vector3(input.x, 0, input.y)).normalized()
			var vetor_x = direction.x * SPEED
			var vetor_z = direction.z * SPEED

			if direction:
				velocity.x = direction.x * SPEED
				velocity.z = direction.z * SPEED
			else:
				velocity.x = move_toward(velocity.x, 0, SPEED)
				velocity.z = move_toward(velocity.z, 0, SPEED)

			if not is_on_floor():
				velocity += get_gravity() * delta

			# Handle jump.
			if Input.is_action_just_pressed("ui_accept") and is_on_floor():
				velocity.y = JUMP_VELOCITY

			# ส่งตำแหน่งของ agent และ target ไปยัง WebSocket
			var stats_Agent = [
				'Agent',
				global_position.x,
				global_position.y,
				global_position.z
				]
			var stats_Target = [
				'Target',
				target.global_position.x,
				target.global_position.y,
				target.global_position.z
				]
			var state_Wall = [
				"Wall",
				wall_1.global_position.x,
				wall_1.global_position.y,
				wall_1.global_position.z,
				
				wall_2.global_position.x,
				wall_2.global_position.y,
				wall_2.global_position.z,
				
				wall_3.global_position.x,
				wall_3.global_position.y,
				wall_3.global_position.z,
				]
			
			var done = 'false'
			var distance = global_position.distance_to(target.global_position)
			var reward = -global_position.distance_to(target.global_position)

			if distance < 2.5:
				done = 'true'
				_reset_position()
			elif distance > 31:
				_reset_position()
				done = 'false'
			else:
				done = 'false'
			var sausage  = [stats_Agent, stats_Target, reward, done, vetor_x, vetor_z,state_Wall,saw_wall]

			socket.send_text(str(sausage))
		# Add epoch count only if the position is updated and message is sent
			epoch_count += 1
			move_and_slide()


func _exit_tree():
	socket.close()
