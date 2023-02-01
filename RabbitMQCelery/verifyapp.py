from flask import Flask
import boto3
from celery import Celery
import pymongo

celery_app = Celery('verifyfacetask', broker='amqp://myuser:password@localhost:5672/myvhost',
                    backend='mongodb://localhost:27017/celerydb')

def call_verify_method(frame, camid, size=(100,100)):
    print('Invoking method.. sending data to verifyfacetask.verifyface')
    res = celery_app.send_task('verifyfacetask.verifyface', kwargs={'frame':frame, 'size':size, 'camid':camid})    #pass queue name.task
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
