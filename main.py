import config
from server.grpc_server import GrpcServer, AIGrpcService

CONFIG_PATH = "configs/config.json"


def main():
    configs = config.read_configs(CONFIG_PATH)

    grpc_configs = configs['grpc']

    grcp_server = GrpcServer(grpc_configs['port'], grpc_configs['workers_count'], AIGrpcService)
    grcp_server.run()


if __name__ == '__main__':
    main()
