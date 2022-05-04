from flask import Flask, redirect, url_for, render_template, request, session
import audio_recorder as ar
import audio_processor as ap

app= Flask(__name__)
app.secret_key = "apple"



@app.route("/", methods=["POST", "GET"])
def home():
    if request.method=="POST":
        ar.audioRecorder()       
        return redirect(url_for("output"))
    else:
        return render_template("home.html")


@app.route("/output", methods=["POST", "GET"])
def output():
    if request.method=="POST":
        user1 = request.form["again"]

        return redirect(url_for("home"))
    else:
        return render_template("output.html",label= ap.labelIdentifier())

if __name__ == "__main__":
    app.run(debug=True)
