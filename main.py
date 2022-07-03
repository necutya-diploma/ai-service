import os
import config
from server.grpc_server import GrpcServer, AIGrpcService
from service.ai import AIService

CONFIG_PATH = "configs/config.json"


def main():
    configs = config.read_configs(CONFIG_PATH)

    base_path = os.path.dirname(os.path.abspath(__file__))

    ai_configs = configs['ai']
    ai_service = AIService(
        model_path=os.path.join(base_path, ai_configs['model_path']),
        tokenizer_path=os.path.join(base_path, ai_configs['tokenizer_path'])
    )

    grpc_configs = configs['grpc']

    ai_grpc_service = AIGrpcService(ai_service)

    grcp_server = GrpcServer(grpc_configs['port'], grpc_configs['workers_count'], ai_grpc_service)

    grcp_server.run()


if __name__ == '__main__':
    main()
