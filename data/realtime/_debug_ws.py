import websocket
import _thread
import time
import json
import logging

def on_message(ws, message):
    print(f"Received message:\n{message}")
    if "CONNECTED" in message:
        print("STOMP Connected! Subscribing to topic...")
        # Subscribe to Delhi Airport RVR topic
        ws.send("SUBSCRIBE\nid:sub-0\ndestination:/topic/rvr/cb167d4b-5c95-4bb4-b4db-cdf74b7e8e8a\n\n\x00")
    if "MESSAGE" in message:
        print("Got RVR data! Closing connection.")
        ws.close()

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
    print("Opened connection")
    # Send STOMP CONNECT frame
    ws.send("CONNECT\naccept-version:1.1,1.0\nheart-beat:10000,10000\n\n\x00")

if __name__ == "__main__":
    websocket.enableTrace(True)
    # Often SockJS websockets have an endpoint like /websocket or similar.
    # We saw /wc/internal/api/websocket/public-subscribe in the logs
    url = "ws://103.215.208.153:8444/wc/internal/api/websocket/public-subscribe"
    
    # Sometimes SockJS adds numbers e.g., /000/random/websocket
    url_sockjs = "ws://103.215.208.153:8444/wc/internal/api/websocket/public-subscribe/websocket"
    
    ws = websocket.WebSocketApp(url_sockjs,
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)

    ws.run_forever()
