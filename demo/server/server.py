#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import msgpack
import json
import argparse

from twisted.internet import reactor, threads
from twisted.web.server import NOT_DONE_YET, Site
from twisted.logger import Logger, textFileLogObserver
from txrestapi.resource import APIResource
from txrestapi.methods import GET, POST
from twisted.web.server import Session

from rmq_interface import RabbitMQInterface


class AsyncServer(APIResource):
    def __init__(self):
        APIResource.__init__(self)

    @staticmethod
    def __generate_pika_connection():
        return RabbitMQInterface(user=config['rabbitmq_user'],
                                 password=config['rabbitmq_pass'],
                                 host=config['rabbitmq_host'],
                                 port=config['rabbitmq_port'])

    @GET(b"^/$")
    def main(self, request):
        try:
            self.__async_handler(request, 'elsa',
                                 {},
                                 'elsa')
            return NOT_DONE_YET
        except Exception:
            log.failure('')
            return 'Wrong data format'

    @POST(b"^/api/v1/run$")
    def summarize(self, request):
        try:
            content = json.loads(request.content.read().decode())
            self.__async_handler(request, 'elsa-summarization',
                                 content,
                                 'elsa-summarization')
            return NOT_DONE_YET
        except Exception:
            log.failure('')
            return 'Wrong data format'

    def __async_handler(self, request, action, payload, routing_key):
        try:
            body = {"action": action, 'payload': payload}
            body = msgpack.packb(body)

            interface = self.__generate_pika_connection()
            result_deferred = threads.deferToThread(interface.fetch, exchange='amq.topic',
                                                    body=body, routing_key=routing_key)
            result_deferred.addCallback(self.__finish_request, request=request)
        except Exception:
            log.failure('')
            return 'Wrong API format'

    @staticmethod
    def __finish_request(data, request, decode=False):
        data = msgpack.unpackb(data)
        if isinstance(data, dict):
            data = json.dumps(data)
        request.write(data)
        request.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='*', help='path to config.json')
    args = parser.parse_args()

    config = json.load(open(args.config[0], 'r'))

    log_file = open(config['server_log'], 'a')
    log = Logger(observer=textFileLogObserver(log_file))

    root = AsyncServer()
    factory = Site(root)

    reactor.listenTCP(config['port'], factory)
    reactor.run()
