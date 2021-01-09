<<<<<<< HEAD
from flask import Flask, render_template, Response
from openCV import VideoCamera

app = Flask(__name__)


@app.route("/")
def index():
    # rendering webpage
    return render_template("index.html")


def gen(camera):
    while True:
        # get camera frame
        frame = camera.get_frame()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(
        gen(VideoCamera()), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    # defining server ip address and port
=======
from flask import Flask, render_template, Response
from openCV import VideoCamera

app = Flask(__name__)


@app.route("/")
def index():
    # rendering webpage
    return render_template("index.html")


def gen(camera):
    while True:
        # get camera frame
        frame = camera.get_frame()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(
        gen(VideoCamera()), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    # defining server ip address and port
>>>>>>> 213a78a9a87989c560404434cfe281ffa1373e11
    app.run(debug=True,threaded=True)