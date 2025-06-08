"""
Research Integrity Warning System
Critical safety measures for academic research use
"""

import warnings
import functools
import os
from typing import Callable, Any

class ResearchIntegrityError(Exception):
    """Raised when research integrity requirements are violated"""
    pass

class ResearchWarning(UserWarning):
    """Warning class for research integrity issues"""
    pass

def research_critical(message: str):
    """
    Decorator to mark functions that are CRITICAL for research integrity
    These functions will fail unless RESEARCH_MODE=strict is set
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            research_mode = os.environ.get('RESEARCH_MODE', 'none')
            
            if research_mode == 'strict':
                raise ResearchIntegrityError(
                    f"\n{'='*80}\n"
                    f"🚨 RESEARCH INTEGRITY VIOLATION 🚨\n"
                    f"{'='*80}\n"
                    f"Function: {func.__name__}\n"
                    f"Issue: {message}\n\n"
                    f"This function has been identified as potentially unreliable for research.\n"
                    f"STRICT MODE: This function is BLOCKED for research integrity.\n"
                    f"To use this function:\n"
                    f"1. Use RESEARCH_MODE=development for testing with warnings\n"
                    f"2. Implement research-grade alternatives per remediation plan\n"
                    f"3. Validate all assumptions against empirical data\n\n"
                    f"For research-grade alternatives, see ACADEMIC_RESEARCH_REMEDIATION_PLAN.md\n"
                    f"{'='*80}\n"
                )
            
            # If not in strict mode, issue warning but allow execution
            warnings.warn(
                f"RESEARCH WARNING - {func.__name__}: {message}",
                ResearchWarning,
                stacklevel=2
            )
            
            return func(*args, **kwargs)
        
        wrapper._research_warning = message
        wrapper._research_status = 'CRITICAL'
        return wrapper
    return decorator

def research_deprecated(message: str, alternative: str = ""):
    """
    Decorator to mark functions deprecated for research use
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            warning_msg = f"DEPRECATED for research: {message}"
            if alternative:
                warning_msg += f" Use {alternative} instead."
            
            warnings.warn(warning_msg, ResearchWarning, stacklevel=2)
            
            return func(*args, **kwargs)
        
        wrapper._research_warning = message
        wrapper._research_status = 'DEPRECATED'
        wrapper._research_alternative = alternative
        return wrapper
    return decorator

def research_requires_validation(empirical_source: str = ""):
    """
    Decorator to mark functions that require empirical validation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not empirical_source:
                warnings.warn(
                    f"{func.__name__} requires empirical validation before research use. "
                    f"No empirical source provided.",
                    ResearchWarning,
                    stacklevel=2
                )
            else:
                warnings.warn(
                    f"{func.__name__} uses parameters from: {empirical_source}. "
                    f"Verify appropriateness for your research context.",
                    ResearchWarning,
                    stacklevel=2
                )
            
            return func(*args, **kwargs)
        
        wrapper._research_empirical_source = empirical_source
        wrapper._research_status = 'REQUIRES_VALIDATION'
        return wrapper
    return decorator

def check_research_mode():
    """Check if proper research mode is set"""
    research_mode = os.environ.get('RESEARCH_MODE', 'none')
    
    if research_mode == 'none':
        print("\n" + "="*80)
        print("⚠️  RESEARCH MODE NOT SET")
        print("="*80)
        print("This codebase contains functions with research integrity issues.")
        print("Before using for academic research, please:")
        print("1. Review ACADEMIC_RESEARCH_REMEDIATION_PLAN.md")
        print("2. Set RESEARCH_MODE environment variable:")
        print("   - 'development' : Allow execution with warnings")
        print("   - 'strict'      : Block risky functions entirely")
        print("3. Consider using research-grade alternatives")
        print("="*80 + "\n")
    
    elif research_mode == 'development':
        print("🔄 RESEARCH MODE: Development (warnings enabled)")
    
    elif research_mode == 'strict':
        print("🔒 RESEARCH MODE: Strict (critical functions blocked)")
    
    else:
        raise ValueError(f"Invalid RESEARCH_MODE: {research_mode}. Use 'development', 'strict', or unset.")

# Export functions for use in other modules
__all__ = [
    'research_critical',
    'research_deprecated', 
    'research_requires_validation',
    'check_research_mode',
    'ResearchIntegrityError',
    'ResearchWarning'
]