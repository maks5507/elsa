#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import rmq_interface
import msgpack


class Worker:

    def __init__(self, action, log):
        self.log = log
        self.action = action

    @rmq_interface.class_consumer
    def __read(self, payload, props):
        data = msgpack.unpackb(payload)

        action_type = data[b'action']
        params = data[b'payload']
        params = {key.decode(): value for key, value in params.items()}

        result = self.action(**params)

        result = msgpack.packb(result)
        return result

    def run(self, rmq_connect, rmq_queue):
        try:
            interface = rmq_interface.RabbitMQInterface(url_parameters=rmq_connect)
            interface.listen(rmq_queue, self.__read)
        except Exception:
            self.log.failure('')
