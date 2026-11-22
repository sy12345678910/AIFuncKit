from exceptions.pipeline_exceptions import CircuitBreakerOpenException

class BreakCircusService:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold: int = failure_threshold
        self.count_errors: int = 0
        self.is_open: bool = False  
    def record_succes(self) -> None:
        "return counter to factories settings"
        self.count_errors = 0
        
    def record_failure(self) -> None:
        "record error and check, is failure threasholder exceeded"
        self.count_errors += 1
        
        if self.count_errors >= self.failure_threshold:
            self.emergency_halt(f"Threashgolder is exexeed ({self.count_errors}/{self.failure_threshold})")
    
    def emergency_halt(self, reason: str = "Alarm stop") -> None:
        """Blocking pipeline"""
        self.is_open = True
        raise CircuitBreakerOpenException(f"Circuit Breaker is stopped: {reason}")
    def check_status(self) -> None:
        """Check, is stopped pipeline"""
        if self.is_open:
            raise CircuitBreakerOpenException("Circuit Breaker is opened: execution is blocked!")
    def reset(self) -> None:
        """Reset threasholder"""
        self.count_errors = 0
        self.is_open = False