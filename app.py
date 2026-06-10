from flask import Flask
from lib.api.loader import loader_bp
from lib.api.eda import eda_bp
from lib.api.cleaner import cleaner_bp
from lib.api.encoder import encoder_bp
from lib.api.splitter import splitter_bp
from lib.api.models import models_bp

app = Flask(__name__)
app.register_blueprint(loader_bp)
app.register_blueprint(eda_bp)
app.register_blueprint(cleaner_bp)
app.register_blueprint(encoder_bp)
app.register_blueprint(splitter_bp)
app.register_blueprint(models_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5001)
