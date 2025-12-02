from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import json

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    fields = db.relationship('Field', backref='owner', lazy=True)

class Field(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    field_type = db.Column(db.String(20), nullable=False)  # polygon, circle, rectangle
    area_hectares = db.Column(db.Float)
    area_acres = db.Column(db.Float)
    geometry_json = db.Column(db.Text, nullable=False)  # Stored as JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.field_type,
            'areaHectares': self.area_hectares,
            'areaAcres': self.area_acres,
            'geometry': json.loads(self.geometry_json),
            'created_at': self.created_at.isoformat()
        }
