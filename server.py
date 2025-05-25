
from flask import Flask , render_template
from Blueprint.Digit_recognizer import recognizer_bp  # import blueprint from recognizer/__init__.py

app = Flask(__name__)

# Register blueprints
app.register_blueprint(recognizer_bp, url_prefix="/recognizer")



# Default route to main page
@app.route("/")
def home():
    return render_template('Home_page.html')


@app.route('/projects')
def projects():
    return render_template('Projects_page.html')


@app.route('/projects/digit_recognizer')
def recognizer():
    return render_template('Digit_recognizer/recognizer.html')


if __name__ == "__main__":
    app.run(debug=True)
