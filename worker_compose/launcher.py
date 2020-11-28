#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import numpy as np
from pathlib import Path
import json
import multiprocessing
from twisted.logger import Logger, textFileLogObserver
import importlib.util

from .worker import Worker
from .processor import Processor


class Launcher:
    def __init__(self, log_file='launcher.log'):
        self.log = Logger(observer=textFileLogObserver(open(log_file, 'a')))

    @staticmethod
    def __split_collection(reference_corpus_path, n_jobs, mask):
        files = [filename for filename in Path(reference_corpus_path).rglob(mask)]
        chunks = np.array_split(files, n_jobs)
        return chunks

    @staticmethod
    def __terminate_all(processes):
        if isinstance(processes, list):
            for process in processes:
                process.terminate()
        else:
            for id in processes:
                for process in processes[id]:
                    process.terminate()

    def launch(self, config):
        processes = {}
        try:
            config = json.loads(config)

            jobs = {}
            started = set()
            for job_id in config:
                jobs[job_id] = config[job_id]
                module_name = jobs[job_id]["name"]
                class_name = ''.join([f'{prt[0].upper()}{prt[1:]}' for prt in module_name.split('_')])

                init_args = config[job_id]['init_args']
                init_args['log'] = self.log

                prefix = jobs[job_id]['prefix']

                spec = importlib.util.spec_from_file_location(module_name, prefix)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                attr = getattr(module, class_name)
                jobs[job_id]['instance'] = attr(**init_args)

            while len(started) != len(jobs):
                for i, job_id in enumerate(jobs):
                    job = jobs[job_id]

                    num_finished = 0
                    for parent_job in job['depends_on']:
                        if parent_job in started and not processes[parent_job].is_alive():
                            num_finished += 1

                    if len(job['depends_on']) == num_finished:
                        started.add(i)
                        if job['mode'] == 'processor':
                            chunks = self.__split_collection(job['texts'], job['n_jobs'], job['mask'])

                        processes[job_id] = []

                        self.log.info('Launching {job_id} number {i}', job_name=job_id, i=i)

                        for j in range(job['n_jobs']):
                            if job['mode'] == 'processor':
                                job['run_args']['chunk'] = chunks[j]
                            if job['add_process_num']:
                                job['run_args']['process_num'] = j

                            target = job['instance'].run
                            if job['mode'] == 'worker':
                                target = Worker(job['instance'].run, self.log).run
                            elif job['mode'] == 'processor':
                                target = Processor(job['instance']).run

                            processes[job_id] += [multiprocessing.Process(target=target,
                                                                          kwargs=job['run_args'])]
                            processes[job_id][-1].start()

                for job_id in processes:
                    for process in processes[job_id]:
                        process.join()

                self.__terminate_all(processes)
        finally:
            self.__terminate_all(processes)
