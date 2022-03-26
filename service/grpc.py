import gen.py.faker_pb2 as pb
import gen.py.faker_pb2_grpc as pb_grpc

from service.ai import Faker


class AIGrpcService(pb_grpc.AIServicer):

    def __init__(self, *args, **kwargs):
        pass

    def CheckMessage(self, request, context):
        res = Faker().check(request.message)
        return self._generate_check_message_response(*res)

    @staticmethod
    def _generate_check_message_response(message: str, is_generated: bool, generated_percent: float):
        resp = {
            "message": message,
            "isGenerated": is_generated,
            "generatedPercent": generated_percent,
        }

        return pb.MessageResponse(**resp)
