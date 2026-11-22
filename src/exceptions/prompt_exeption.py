from exceptions.exceptions import __AIBaseException

class __BasePromptException(__AIBaseException):
    pass

class PromptInjectionError(__BasePromptException):
    def __init__(self, message: str):
        super().__init__(message)
        
class TooLongPromptError(__BasePromptException):
    def __init__(self, message: str):
        super().__init__(message)
        