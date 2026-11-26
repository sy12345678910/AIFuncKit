from aifunckit.exceptions.pipeline_exceptions import CircuitBreakerOpenException
from aifunckit._internal.log_config import logger

class BreakCircusService:
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold: int = failure_threshold
        self.count_errors: int = 0
        self.is_open: bool = False  

    def record_succes(self) -> None:
        "return counter to factories settings"
        if self.count_errors > 0:
            logger.debug("Circuit breaker resetting error count after success")
        self.count_errors = 0

    record_success = record_succes
        
    def record_failure(self) -> None:
        "record error and check, is failure threasholder exceeded"
        self.count_errors += 1
        logger.warning(
            "Circuit breaker recorded failure (%d/%d)",
            self.count_errors,
            self.failure_threshold,
        )
        
        if self.count_errors >= self.failure_threshold:
            self.emergency_halt(f"Threashgolder is exexeed ({self.count_errors}/{self.failure_threshold})")
    
    def emergency_halt(self, reason: str = "Alarm stop") -> None:
        """Blocking pipeline"""
        self.is_open = True
        logger.error("Circuit breaker emergency halt triggered: %s", reason)
        raise CircuitBreakerOpenException(f"Circuit Breaker is stopped: {reason}")

    def check_status(self) -> None:
        """Check, is stopped pipeline"""
        if self.is_open:
            logger.warning("Circuit breaker check blocked execution (breaker is open)")
            raise CircuitBreakerOpenException("Circuit Breaker is opened: execution is blocked!")

    def reset(self) -> None:
        """Reset threasholder"""
        self.count_errors = 0
        self.is_open = False
        logger.info("Circuit breaker has been reset and closed")