# the script is used to create the database
import sqlite3
from main import create_app
from db import db

connection = sqlite3.connect('database.db')

with open('schema.sql') as f:
    connection.executescript(f.read())

connection.commit()
connection.close()

db.create_all(app=create_app())
