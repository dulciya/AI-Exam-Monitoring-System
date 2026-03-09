# dashboard.py
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import base64, os, time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/alert', methods=['POST'])
def receive_alert():
    data = request.get_json()
    if not data:
        return jsonify({"ok": False}), 400

    if "frame_b64" in data:
        b = base64.b64decode(data["frame_b64"])
        fname = f"alerts/{int(time.time()*1000)}.jpg"
        os.makedirs("alerts", exist_ok=True)
        with open(fname, "wb") as f:
            f.write(b)
        data["thumbnail"] = fname
        data.pop("frame_b64", None)

    socketio.emit('new_alert', data, broadcast=True)
    return jsonify({"ok": True})

if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
