extends CharacterBody3D

const SPEED = 5.0
const JUMP_VELOCITY = 4.5

var socket = WebSocketPeer.new()
var has_sent = false


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
	
	if state == WebSocketPeer.STATE_OPEN:
		socket.send_text("")
		
		if not has_sent:
			socket.send_text("Hello Worldfrom Godot!")
			has_sent = true
			print("Sent message to server!")
			
			
			
		while socket.get_available_packet_count() > 0:
			var message = socket.get_packet().get_string_from_utf8()
			if message:
				print("Received message from server: ", message)
			
	elif state == WebSocketPeer.STATE_CLOSED:
		print("Connection closed.")
		set_process(false)

func _physics_process(delta: float) -> void:
	if position.y < -1 :
		position.x = 0
		position.y = 2
		position.z = 0
	
	if not is_on_floor():
		velocity += get_gravity() * delta

	if Input.is_action_just_pressed("key_Spacebar") and is_on_floor():
		velocity.y = JUMP_VELOCITY 
		
	var input_dir := Input.get_vector("key_A", "key_D", "key_W", "key_S")
	var direction := (transform.basis * Vector3(input_dir.x, 0, input_dir.y)).normalized()
	if direction:
		velocity.x = direction.x * SPEED
		velocity.z = direction.z * SPEED
	else:
		velocity.x = move_toward(velocity.x, 0, SPEED)
		velocity.z = move_toward(velocity.z, 0, SPEED)

	move_and_slide()

func _exit_tree():
	socket.close()
