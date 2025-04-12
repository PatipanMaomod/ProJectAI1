extends Node

var socket = WebSocketPeer.new()
var has_sent = false  # ตัวแปรเพื่อเช็คว่าส่งข้อความไปหรือยัง

func _ready():
	var url = "ws://localhost:8765"
	var error = socket.connect_to_url(url)
	
	if error == OK:
		print("Connected to server!")
	else:
		print("Failed to connect: ", error)

func _process(delta):
	var x = delta
	x = x+1
	socket.poll()  # อัปเดตสถานะของ socket
	var state = socket.get_ready_state()
	
	if state == WebSocketPeer.STATE_OPEN:
		if not has_sent:
			socket.send_text("Hello Worldfrom Godot!")
			has_sent = true
			print("Sent message to server!")
			
		if Input.is_action_just_pressed("key_A"):
			socket.send_text("A")
		
		
		
	
		
		# อ่านข้อความจากเซิร์ฟเวอร์
		while socket.get_available_packet_count() > 0:
			var message = socket.get_packet().get_string_from_utf8()
			if message:
				print("Received message from server: ", message)
		
				
	elif state == WebSocketPeer.STATE_CLOSED:
		print("Connection closed.")
		set_process(false)  # หยุด _process เมื่อหลุดการเชื่อมต่อ

func _exit_tree():
	socket.close() 
