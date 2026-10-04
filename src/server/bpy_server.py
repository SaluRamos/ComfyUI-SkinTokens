from bottle import request, response

import bottle
import os
import queue
import threading
import traceback

from .spec import bytes_to_object, object_to_bytes, BPY_PORT
from .codec import MAX_PACKET_BYTES
from .transport import authenticated, resolve_payload_path, server_token, payload_directory

from ..rig_package.parser.bpy import BpyParser, transfer_rigging


def _resolve_payload(data):
    if isinstance(data, dict) and "payload_path" in data:
        payload_path = resolve_payload_path(data["payload_path"])
        with open(payload_path, "rb") as f:
            if os.fstat(f.fileno()).st_size > MAX_PACKET_BYTES:
                raise ValueError("Payload exceeds 2 GiB")
            return bytes_to_object(f.read(MAX_PACKET_BYTES + 1))
    return data

def create_app(path_queue, result_queue):
    app = bottle.Bottle()

    @app.hook('before_request')
    def authorize():
        if request.headers.get('Origin') or not authenticated(request.headers.get('Authorization')):
            bottle.abort(403, "Unauthorized SkinTokens request")
        response.set_header('X-SkinTokens-Protocol', '1')
        if request.content_length > MAX_PACKET_BYTES:
            bottle.abort(413, "Payload exceeds 2 GiB")
    
    @app.route('/load', method='GET') # type: ignore
    def load():
        data = request.body.read() # type: ignore
        path_queue.put(('load', data))
        res = result_queue.get()
        payload = object_to_bytes(res)
        response.content_type = 'application/octet-stream'  # type: ignore
        return payload
    
    @app.route('/ping', method='GET') # type: ignore
    def ping():
        return 'pong'
    
    @app.route('/export', method='post') # type: ignore
    def export():
        data = request.body.read() # type: ignore
        path_queue.put(('export', data))
        res = result_queue.get()
        payload = object_to_bytes(res)
        response.content_type = 'application/octet-stream'  # type: ignore
        return payload
    
    @app.route('/transfer', method='post') # type: ignore
    def transfer():
        data = request.body.read() # type: ignore
        path_queue.put(('transfer', data))
        res = result_queue.get()
        payload = object_to_bytes(res)
        response.content_type = 'application/octet-stream'  # type: ignore
        return payload

    return app


def run():
    server_token()
    payload_directory()
    path_queue = queue.Queue()
    result_queue = queue.Queue()
    app = create_app(path_queue, result_queue)
    
    def run_server(): bottle.run(app, host='127.0.0.1', port=BPY_PORT, server='tornado')
    threading.Thread(target=run_server, daemon=False).start()
    
    while True:
        d = path_queue.get()
        op = d[0]
        try:
            data = _resolve_payload(bytes_to_object(d[1]))
            if op == 'load':
                print("[SERVER] received load path:", data)
                asset = BpyParser.load(data)
                result_queue.put(asset)
            elif op == 'export':
                print("[SERVER] received export path:", data['filepath'])
                BpyParser.export(**data)
                result_queue.put('ok')
            elif op == 'transfer':
                print("[SERVER] received transfer path:", data['target_path'])
                transfer_rigging(**data)
                result_queue.put('ok')
            else:
                result_queue.put(f"unsupported op: {str(op)}")
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            result_queue.put({
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb,
            })
