"""
Enhanced exception types for the project analyzer system.
Provides specific, actionable exception types for better error handling and debugging.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ProjectAnalyzerError(Exception):
    """Base exception for all project analyzer errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        logger.error(
            f"ProjectAnalyzerError: {message}", extra={"details": self.details}
        )


class ConfigurationError(ProjectAnalyzerError):
    """Configuration-related errors."""

    pass


class ResourceValidationError(ProjectAnalyzerError):
    """System resource validation errors."""

    def __init__(self, resource_type: str, current: Any, required: Any, **kwargs):
        self.resource_type = resource_type
        self.current = current
        self.required = required
        message = (
            f"Insufficient {resource_type}: current={current}, required={required}"
        )
        super().__init__(message, **kwargs)


class MemoryLimitError(ResourceValidationError):
    """Insufficient memory for operation."""

    def __init__(self, current_mb: float, required_mb: float, **kwargs):
        self.current_mb = current_mb
        self.required_mb = required_mb
        super().__init__("memory", current_mb, required_mb, **kwargs)


class DiskSpaceError(ResourceValidationError):
    """Insufficient disk space for operation."""

    def __init__(self, current_mb: float, required_mb: float, path: str, **kwargs):
        self.current_mb = current_mb
        self.required_mb = required_mb
        self.path = path
        super().__init__("disk_space", current_mb, required_mb, **kwargs)


class FileAnalysisError(ProjectAnalyzerError):
    """File analysis specific errors."""

    def __init__(self, file_path: str, error_type: str, **kwargs):
        self.file_path = str(file_path)
        self.error_type = error_type
        message = f"Analysis failed for {file_path}: {error_type}"
        super().__init__(message, **kwargs)


class BinaryFileError(FileAnalysisError):
    """Binary file detected during text analysis."""

    def __init__(self, file_path: str, file_size: int = 0, **kwargs):
        self.file_size = file_size
        message = f"Binary file detected: {file_path} (size: {file_size} bytes)"
        super().__init__(file_path, "binary_file", **kwargs)


class EncodingError(FileAnalysisError):
    """File encoding related errors."""

    def __init__(self, file_path: str, encoding_attempted: str = "utf-8", **kwargs):
        self.encoding_attempted = encoding_attempted
        message = f"Encoding error for {file_path} with {encoding_attempted}"
        super().__init__(file_path, "encoding_error", **kwargs)


class ParsingError(FileAnalysisError):
    """Syntax or parsing errors in source files."""

    def __init__(
        self,
        file_path: str,
        line_number: Optional[int] = None,
        syntax_details: Optional[str] = None,
        **kwargs,
    ):
        self.line_number = line_number
        self.syntax_details = syntax_details
        details = kwargs.get("details", {})
        if line_number is not None:
            details["line_number"] = line_number
        if syntax_details:
            details["syntax_details"] = syntax_details
        kwargs["details"] = details

        message = f"Parsing error in {file_path}"
        if line_number:
            message += f" at line {line_number}"
        if syntax_details:
            message += f": {syntax_details}"
        super().__init__(file_path, "parsing_error", **kwargs)


class ModelError(ProjectAnalyzerError):
    """AI model related errors."""

    def __init__(self, model_name: str, error_type: str, **kwargs):
        self.model_name = model_name
        self.error_type = error_type
        message = f"Model error [{model_name}]: {error_type}"
        super().__init__(message, **kwargs)


class EmbeddingGenerationError(ModelError):
    """Embedding generation specific errors."""

    def __init__(self, model_name: str, file_count: int = 0, **kwargs):
        self.file_count = file_count
        message = f"Embedding generation failed for {model_name} (files: {file_count})"
        super().__init__(model_name, "embedding_generation", **kwargs)


class TrackerUpdateError(ProjectAnalyzerError):
    """Tracker file update errors."""

    def __init__(self, tracker_path: str, operation: str, **kwargs):
        self.tracker_path = str(tracker_path)
        self.operation = operation
        message = f"Tracker update failed for {tracker_path} during {operation}"
        super().__init__(message, **kwargs)


class StateManagementError(ProjectAnalyzerError):
    """State management and backup/restore errors."""

    def __init__(self, operation: str, **kwargs):
        self.operation = operation
        message = f"State management error during {operation}"
        super().__init__(message, **kwargs)


class CacheError(ProjectAnalyzerError):
    """Cache-related errors."""

    def __init__(self, cache_name: str, operation: str, **kwargs):
        self.cache_name = cache_name
        self.operation = operation
        message = f"Cache error [{cache_name}] during {operation}"
        super().__init__(message, **kwargs)


class ValidationError(ProjectAnalyzerError):
    """Data validation errors."""

    def __init__(self, field_name: str, value: Any, expected: str, **kwargs):
        self.field_name = field_name
        self.value = value
        self.expected = expected
        message = f"Validation error: {field_name}={value} (expected: {expected})"
        super().__init__(message, **kwargs)


class PathError(ProjectAnalyzerError):
    """Path-related errors."""

    def __init__(self, path: str, error_type: str, **kwargs):
        self.path = str(path)
        self.error_type = error_type
        message = f"Path error [{path}]: {error_type}"
        super().__init__(message, **kwargs)


class PermissionError(ProjectAnalyzerError):
    """File/directory permission errors."""

    def __init__(self, path: str, operation: str, **kwargs):
        self.path = str(path)
        self.operation = operation
        message = f"Permission denied for {path} during {operation}"
        super().__init__(message, **kwargs)


class NetworkError(ProjectAnalyzerError):
    """Network-related errors."""

    def __init__(self, url: str, operation: str, **kwargs):
        self.url = url
        self.operation = operation
        message = f"Network error for {url} during {operation}"
        super().__init__(message, **kwargs)


class TimeoutError(ProjectAnalyzerError):
    """Operation timeout errors."""

    def __init__(self, operation: str, timeout_seconds: float, **kwargs):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        message = f"Operation '{operation}' timed out after {timeout_seconds} seconds"
        super().__init__(message, **kwargs)


# Error recovery helpers
def handle_file_analysis_error(
    file_path: str, original_error: Exception
) -> FileAnalysisError:
    """Convert various file-related exceptions to FileAnalysisError."""

    if isinstance(original_error, PermissionError):
        return PermissionError(file_path, "read")
    elif isinstance(original_error, UnicodeDecodeError):
        return EncodingError(file_path, str(original_error.encoding))
    elif isinstance(original_error, SyntaxError):
        return ParsingError(
            file_path,
            original_error.lineno,
            str(original_error),
            details={
                "filename": original_error.filename,
                "offset": original_error.offset,
            },
        )
    else:
        # Generic file analysis error with original exception details
        return FileAnalysisError(
            file_path,
            f"unhandled_error: {type(original_error).__name__}",
            details={
                "original_error": str(original_error),
                "error_type": type(original_error).__name__,
            },
        )


def handle_model_error(original_error: Exception, model_name: str) -> ModelError:
    """Convert various model-related exceptions to ModelError."""

    error_type = f"{type(original_error).__name__}: {str(original_error)}"
    return ModelError(model_name, error_type)


def log_and_reraise(
    logger_instance: logging.Logger,
    error: Exception,
    context: str = "",
    reraise: bool = True,
) -> Optional[Exception]:
    """Log error and optionally re-raise with enhanced context."""

    if context:
        logger_instance.error(f"Error in {context}: {error}")
    else:
        logger_instance.error(f"Error: {error}")

    if isinstance(error, ProjectAnalyzerError):
        # Already logged and enhanced
        if reraise:
            raise error
        return error
    else:
        # Wrap unknown errors
        enhanced_error = ProjectAnalyzerError(
            f"{type(error).__name__}: {str(error)}",
            details={"original_type": type(error).__name__, "context": context},
        )
        if reraise:
            raise enhanced_error from error
        return enhanced_error
