from flask import Flask, request, jsonify   # core objects we’ll use

app = Flask(__name__)                       # the WSGI application object

@app.route("/")                              # a *route* (URL pattern)
def hello_world():
    return "Hello, Flask!"

if __name__ == "__main__":                  # only true when run directly
    app.run(debug=True, port=5000)          # built‑in dev server
