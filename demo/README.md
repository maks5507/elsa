# ELSA demo

The demo is avaliable at http://elsa-demo.com

## Launching demo stand

```bash
docker pull rabbitmq

docker run -it -d \
           --name rabbitmq \
	         -p 15672:15672 \
	         -p 5672:5672 \
	         -e RABBITMQ_DEFAULT_USER=guest \
 	         -e RABBITMQ_DEFAULT_PASS=guest \
	         rabbitmq:3-management

docker-compose build
docker-compose up
```

The demo will be avaliable then at http://localhost:22556

## Configuring demo

Before launching the demo, you have to specify models' `prefix`, `init_args` and `run_args` in `config.json` file. 

* `summarization_worker.prefix` -- full path to the `summarization_worker.py`
* `summarization_worker.init_args` -- dict of ELSAs' models init parameters
* `summarization_worker.run_args.rmq_connect` -- connection string for the RMQ (example: `amqp://guest:guest@localhost:5672`)
* `summarization_worker.run_args.rmq_queue` -- RMQ queue for `summarization_worker` to listen
*  `renderer_worker.init_args.main_template_path` -- full path to the `main.html` 
* `renderer_worker.init_args.models` -- list of models' ids specified in `summarization_worker.init_args.list_elsa_params`
* `renderer_worker.init_args.models_values` -- mapping of models' ids to models' names (models' names will appear on a demo webpage)
* `renderer_worker.run_args.rmq_connect` -- same as `summarization_worker.run_args.rmq_connect`
* `renderer_worker.run_args.rmq_queue` -- RMQ queue for `renderer_worker` to listen