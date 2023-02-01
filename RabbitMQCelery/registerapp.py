from flask import Flask
import boto3
from celery import Celery
import pymongo

#app = Flask(__name__)

celery_app = Celery('registerfacetask', broker='amqp://myuser:password@localhost:5672/myvhost',
                    backend='mongodb://localhost:27017/celerydb')

#@app.route('/simple_start_task/<frame>')
def call_register_method(frame, camid, size=(100,100)):
    print('Invoking method.. sending data to registerfacetask.registerface')
    res = celery_app.send_task('registerfacetask.registerface', kwargs={'frame':frame, 'size':size, 'camid':camid})    #pass queue name.task
    print(res.backend)
    return res.id

#@app.route('/simple_task_status/<task_id>')
def get_status(task_id):
    status = celery_app.AsyncResult(task_id, app=celery_app)
    print('Invoking method...')
    return status.state

#@app.route('/simple_task_result/<task_id>')
def task_result(task_id):
    result = celery_app.AsyncResult(task_id).result
    return result
