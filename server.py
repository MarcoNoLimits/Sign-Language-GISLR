import asyncio
import websockets
import json
import uuid
import time
from extractor import WebcamStream
from buffer import SequenceBuffer
from model_engine import ModelEngine

class SignLanguageServer:
    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = port
        self.clients = set()
        
        self.stream = WebcamStream()
        self.buffer = SequenceBuffer(target_frames=100)
        self.model = ModelEngine()
        
        self.running = False

    async def register(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)

    async def broadcast(self, message):
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )

    async def inference_loop(self):
        print("Starting Inference Loop...")
        import cv2
        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic

        while self.running:
            # Sleep small amount to allow other async tasks
            await asyncio.sleep(0.01)
            
            landmarks, frame, results = self.stream.get_frame_data()
            if frame is None:
                continue
                
            self.buffer.add_frame(landmarks)
            
            # -- UI Debug Display --
            image = frame.copy()
            if results:
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                
            cv2.imshow('PC Local Test: Sign Language Server', image)
            if cv2.waitKey(1) & 0xFF == 27: # ESC key to exit window
                self.running = False
                break
            
            # Run inference every ~30 frames
            if self.buffer.is_ready(min_frames=30):
                tensor = self.buffer.get_tensor()
                predicted_word = self.model.predict(tensor)
                
                if predicted_word:
                    print(f"Predicted: {predicted_word}")
                    payload = json.dumps({
                        "id": str(uuid.uuid4()),
                        "text": predicted_word,
                        "timestamp": time.time()
                    })
                    await self.broadcast(payload)
                    self.buffer.clear() # Clear after successful prediction

    async def start(self):
        self.running = True
        print(f"Server starting on ws://{self.host}:{self.port}")
        
        # Start the WebSocket server
        server = await websockets.serve(self.register, self.host, self.port)
        
        # Start the background inference loop
        inference_task = asyncio.create_task(self.inference_loop())
        
        await asyncio.gather(server.wait_closed(), inference_task)

if __name__ == "__main__":
    server = SignLanguageServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("Shutting down...")
        server.stream.release()
