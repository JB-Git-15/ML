from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def load_page():
    return('Predict the next')
app.run()
