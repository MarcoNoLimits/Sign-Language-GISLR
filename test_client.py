import asyncio
import websockets
import json

async def receive_translations():
    uri = "ws://localhost:8080"
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to Sign Language HoloLens Server!")
            print("Listening for translations...")
            
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                print(f"--> Received Translation: {data['text']}")
    except ConnectionRefusedError:
        print(f"Could not connect to server at {uri}. Is it running?")

if __name__ == "__main__":
    asyncio.run(receive_translations())
