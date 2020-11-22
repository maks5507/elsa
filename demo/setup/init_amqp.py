#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import argparse
import json

from .. import RabbitMQInterface


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs='*', help='path to config.json')
    args = parser.parse_args()

    config = json.load(open(args.config[0], 'r'))

    interface = RabbitMQInterface(user=config['rabbitmq_user'],
                                  password=config['rabbitmq_pass'],
                                  host=config['rabbitmq_host'],
                                  port=config['rabbitmq_port'])

    interface.__create_queue(name=config['rabbitmq_queue'],
                             exchange_to_bind='amq.topic',
                             binding_routing_key=config['rabbitmq_queue'])

