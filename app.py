from flask import Flask, render_template, request, jsonify
from chatbot import get_bot_response

app = Flask(__name__)

# Home route → loads frontend UI
@app.route("/")
def index():
    return render_template("index.html")


# Chat API → called from frontend using fetch()
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    # Safety check
    if not data or "message" not in data:
        return jsonify({"reply": "Invalid request"}), 400

    user_message = data["message"]

    # Get AI response
    bot_reply = get_bot_response(user_message)

    return jsonify({"reply": bot_reply})


# Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

