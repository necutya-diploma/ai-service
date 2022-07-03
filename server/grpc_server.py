import grpc

import gen.py.faker_pb2_grpc as pb_grpc

from typing import Type
from concurrent import futures
from service.grpc import AIGrpcService


class GrpcServer:
    def __init__(self, port: str, workers_count: int, ai_service: AIGrpcService, *args, **kwargs):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers_count))
        self.server.add_insecure_port(port)

        pb_grpc.add_AIServicer_to_server(ai_service, self.server)

        self.port = port

    def run(self):
        self.server.start()
        self.server.wait_for_termination()
        print(f'Starting grpc server: addr=localhost:{self.port}')
