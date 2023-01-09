import logging
from logging.handlers import QueueHandler, QueueListener
import zmq

###
### ZeroMQ multi-process logging
###

logger: logging.Logger = logging.getLogger('unify')
LOG_SOCKET_PORT = 5559

##
## The handler runs in the *dammon* process and listens for log messages to dispatch over ZeroMQ
##
class ZeroMQSocketHandler(QueueHandler):
    def __init__(self):
        self.ctx = zmq.Context()
        socket = self.ctx.socket(zmq.PUSH)
        socket.connect(f"tcp://localhost:{LOG_SOCKET_PORT}")
        super().__init__(socket)
        logger.debug("Started ZeroMQSocketHandler")
        self._job = None

    def set_job(self, job):
        self._job = job

    def enqueue(self, record):
        if self._job:
            record.__dict__.update(self._job.__dict__)
        self.queue.send_json(record.__dict__)

    def close(self):
        self.queue.close()

##
## The listener runs in the command process and forwards messages received to the local logging system
##
class ZeroMQSocketListener(QueueListener):
    def __init__(self, *handlers):
        self.ctx = zmq.Context()
        socket = zmq.Socket(self.ctx, zmq.PULL)
        socket.bind(f"tcp://127.0.0.1:{LOG_SOCKET_PORT}")
        super().__init__(socket, *handlers)

    def dequeue(self, block):
        msg = self.queue.recv_json()
        record = logging.makeLogRecord(msg)
        #print(f"[*** {record.msg}] Zeromq logging record ", record)
        return record        


