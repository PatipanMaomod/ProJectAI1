import asyncio
import websockets

async def echo(websocket):
    try:
        while True:
            message = await websocket.recv()
            if message:
                print(f"Received message: {message}")

            user_input = await asyncio.to_thread(input, "Enter message to send: ")
            await websocket.send(user_input)

    except websockets.ConnectionClosed:
        print("Client disconnected.")

async def main():
    start_server = websockets.serve(echo, "localhost", 8765)
    await start_server
    print("Server started at ws://localhost:8765")

    await asyncio.Future()


asyncio.run(main())