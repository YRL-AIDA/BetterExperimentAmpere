from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ApiResult:
    api_success: bool = False
    raw_response: str = ""
    reasoning: str = ""
    parse_success: bool = False
    parsed_headers: List[Dict[str, Any]] = field(default_factory=list)
    parse_error: str = ""
    output_complete: bool = False
    duration_sec: Optional[float] = None
    retry_attempts: int = 0
    requested_max_tokens: int = 0
    effective_max_tokens: int = 0
    budget_clamped: bool = False
    capped: bool = False
    continuation_used: bool = False
    continuation_tier: int = 0
    continuation_forced: bool = False
    tokens_used: Optional[Dict[str, int]] = None
    error_type: str = ""
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_success": self.api_success,
            "raw_response": self.raw_response,
            "reasoning": self.reasoning,
            "parse_success": self.parse_success,
            "parsed_headers": self.parsed_headers,
            "parse_error": self.parse_error,
            "output_complete": self.output_complete,
            "duration_sec": self.duration_sec,
            "retry_attempts": self.retry_attempts,
            "requested_max_tokens": self.requested_max_tokens,
            "effective_max_tokens": self.effective_max_tokens,
            "budget_clamped": self.budget_clamped,
            "capped": self.capped,
            "continuation_used": self.continuation_used,
            "continuation_tier": self.continuation_tier,
            "continuation_forced": self.continuation_forced,
            "tokens_used": self.tokens_used,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }
