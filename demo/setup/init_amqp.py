#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import argparse

from rmq_interface import RabbitMQInterface


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-rq', '--renderer_queue', nargs='*', help='queue name for renderer')
    parser.add_argument('-dq', '--demo_queue', nargs='*', help='queue name for summarization worker')
    parser.add_argument('-c', '--connect', nargs='*', help='queue name for summarization worker')
    args = parser.parse_args()

    print(args.demo_queue)

    interface = RabbitMQInterface(url_parameters=args.connect[0])

    interface.create_queue(name=args.renderer_queue[0],
                           exchnage_to_bind='amq.topic',
                           binding_routing_key=args.renderer_queue[0])

    interface.create_queue(name=args.demo_queue[0],
                           exchnage_to_bind='amq.topic',
                           binding_routing_key=args.demo_queue[0])

