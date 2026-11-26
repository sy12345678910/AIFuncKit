class BaseDBError(Exception):
    pass

class DBConnectionError(BaseDBError):
    def __init__(self, message):
        super().__init__(message)
        
class DBArchiveError(BaseDBError):
    def __init__(self, *args):
        super().__init__(*args)