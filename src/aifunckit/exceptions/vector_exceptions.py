class __VectorException(Exception):
    pass

class VectorGenerationError(__VectorException):
    def __init__(self, message="Vector isn't generated correctly"):
        super().__init__(message)
        
class FetchVectorError(__VectorException):
    def __init__(self, *args):
        super().__init__(*args)