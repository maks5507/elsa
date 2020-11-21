#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import json
import argparse

from twisted.internet import reactor, threads
from twisted.web.server import NOT_DONE_YET, Site
from twisted.logger import Logger, textFileLogObserver
from .txrestapi.resource import APIResource
from .txrestapi.methods import GET
from twisted.web.server import Session

from .. import rmq_interface


class AsyncServer(APIResource):
    def __init__(self):
        APIResource.__init__(self)

    @staticmethod
    def __generate_pika_connection():
        return rmq_interface.RabbitMQInterface(user=config['rabbitmq_user'],
                                               password=config['rabbitmq_pass'],
                                               host=config['rabbitmq_host'],
                                               port=config['rabbitmq_port'])

    def __insert_session(self, request, user_id):
        session = request.getSession()
        uid = session.uid
        session.notifyOnExpire(self._expired(uid))
        self.sessions[uid] = {'user_id': user_id, 'valid': True}

    def __is_expired(self, uid):
        def delete_session():
            del self.sessions[uid]
        return delete_session

    def __check_login(self, request):
        uid = request.getSession().uid
        if uid in self.sessions and self.sessions[uid]['valid']:
            return self.sessions[uid]['user_id']
        return None

    @GET(b"^/$")
    def main(self, request):
        try:
            content = json.loads(request.content.read().decode())
            self.__async_handler(request, 'main',
                                 content,
                                 config['rabbitmq_queue'])
            return NOT_DONE_YET
        except Exception:
            log.failure('')
            return 'Wrong data format'

    def __async_handler(self, request, action, payload, routing_key):
        try:
            body = {"action": action, 'payload': payload}
            body = json.dumps(body)

            interface = self.__generate_pika_connection()
            result_deferred = threads.deferToThread(interface.fetch, exchange='amq.topic',
                                                    body=body, routing_key=routing_key)
            result_deferred.addCallback(self.__finish_request, request=request)
        except Exception:
            log.failure('')
            return 'Wrong API format'

    @staticmethod
    def __finish_request(data, request):
        request.write(data)
        request.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='*', help='path to config.json')
    args = parser.parse_args()

    config = json.load(open(args.config[0], 'r'))

    log_file = open(config['server_log'], 'a')
    log = Logger(observer=textFileLogObserver(log_file))

    class CustomSession(Session):
        sessionTimeout = config['session_timeout']

    root = AsyncServer()
    factory = Site(root)
    factory.sessionFactory = CustomSession

    reactor.listenTCP(config['port'], factory)
    reactor.run()
