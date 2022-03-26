class Faker:

    def __init__(self, *args, **kwargs):
        pass

    def check(self, message) -> (str, bool, float):
        is_generated = True
        generated_percent = 70.
        return message, is_generated, generated_percent
