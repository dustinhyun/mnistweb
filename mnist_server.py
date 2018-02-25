#!/usr/bin/env python

from __future__ import print_function

import sys
import threading
import urllib
import numpy
import tensorflow as tf

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer

# To communicate with TensorFlow server.
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

# Toy web server to get inference requests from the browser.
class Server(BaseHTTPRequestHandler):

    channel = None
    stub = None
    tfs = "localhost"
    tfs_port = 8080
    prediction = ""

    def do_HEAD(self):
        self._setHeader()
        
    def do_GET(self):

        self._setHeader()

        filepath = "mnist.html"
        if self.path.find("jquery-3.3.1.min.js") > 0:
            filepath = "jquery-3.3.1.min.js"

        fr = open(filepath, "r")
        self.wfile.write(fr.read())
        fr.close()

    def do_POST(self):

        length = int(self.headers['Content-Length'])
        data = self._getX(self.rfile.read(length))

        # Sends inference request to your server.
        res = self._inference(data)

        self._setHeader()
        self.wfile.write(res)
        
    def _setHeader(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def _getX(self, data):
        raw = urllib.unquote(data).split("=")[1].split(",")
        data = map(lambda x: float(x), raw)
        return data

    def _inference(self, data):

        # Lazy connection to your TensorFlow server.
        if Server.channel is None:
            Server.channel = implementations.insecure_channel(
                Server.tfs,
                Server.tfs_port
            )

        if Server.stub is None:
            Server.stub = prediction_service_pb2.beta_create_PredictionService_stub(
                Server.channel
            )
        
        tfsReq = predict_pb2.PredictRequest()
        tfsReq.model_spec.name = 'mnist'
        tfsReq.model_spec.signature_name = 'predict_images'
        tfsReq.inputs['images'].CopyFrom(
            tf.contrib.util.make_tensor_proto(data, shape=[1, len(data)])
        )
        future = Server.stub.Predict.future(tfsReq, 5.0)
        result = future.result().outputs['scores'].float_val
        prediction = numpy.argmax(result)
        print("Predicted: %s" % prediction)

        return prediction

def run(server_class=HTTPServer, port=80, tfs="localhost", tfs_port=8080):
    
    Server.tfs, Server.tfs_port = tfs, tfs_port

    server_conf = ('', port)
    httpd = server_class(server_conf, Server)

    print('Starting httpd at %d...' % port)

    httpd.serve_forever()

if __name__ == "__main__":

    argn = len(sys.argv)

    if argn != 3:
        print('Usage: ./mnist_server.py <TFS_IP_OR_HOST>:<TFS_PORT> <PORT>')
    else:
        tfs_conf = sys.argv[1].split(':')[:2]
        run(tfs=tfs_conf[0], tfs_port=int(tfs_conf[1]), port=int(sys.argv[2]))
