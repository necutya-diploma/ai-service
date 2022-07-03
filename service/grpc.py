import gen.py.faker_pb2 as pb
import gen.py.faker_pb2_grpc as pb_grpc


class AIGrpcService(pb_grpc.AIServicer):

    def __init__(self, ai_service, *args, **kwargs):
        self.ai_service = ai_service

    def CheckMessage(self, request, context):
        res = self.ai_service.check_message(request.message)
        return self._generate_check_message_response(*res)

    @staticmethod
    def _generate_check_message_response(message: str, is_generated: bool, generated_percent: float):
        resp = {
            "message": message,
            "isGenerated": is_generated,
            "generatedPercent": generated_percent,
        }

        return pb.MessageResponse(**resp)
