from flask import Flask
app = Flask(__name__)


@app.route('/')
def hello_world():
    return "<h1>Hello World from Containerized App</h1>"


if __name__ == "__main__":
    app.run(debug=True)
