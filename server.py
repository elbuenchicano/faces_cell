from flask import Flask
from flask import Flask
from flask import jsonify
from engine import *
from    flask import request
import  json

from faces import FaceRecognition

################################################################################
################################################################################
################################################################################
app = Flask(__name__)

@app.route("/")
def hello():
    return "<h1 style='color:blue'> Server Faces de Raji!</h1>"

################################################################################
################################################################################
################################################################################
@app.route("/get")
def get():
    return jsonify(queryStudents())

################################################################################
################################################################################
################################################################################
@app.route("/getpost", methods=['GET','POST'])
def post():
    data = {"process": 'ok_update'}
    if request.method == 'GET':
    	data = updateStudents(json.loads(request.data))
    	return jsonify(data)
    if request.method == 'POST':
        data = updateStudents(json.loads(request.data))
        return jsonify(data)

################################################################################
################################################################################
################################################################################
@app.route("/getpost_image", methods=['GET','POST'])
def postImage():
    data = {"process": 'ok_image'}
    if request.method == 'GET':
        data = queryImage(json.loads(request.data))

    if request.method == 'POST':
        data = queryImage(json.loads(request.data))

    return jsonify(data)

################################################################################
################################################################################
################################################################################
if __name__ == "__main__":
    #facer.getDbEmbeddings()

    app.run(host='192.168.0.19')
    
    #updateStudents([{'id':1, 'presente': True}])

    #queryStudents()
    #facer = FaceRecognition()
    #facer.validation_()
    pass 