#!/usr/bin/env python3
"""
Lumii Voice Assistant - Raspberry Pi 5 Optimized Version
Main entry point module
"""

import os
from assistant import LumiiAssistant

# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point with optimizations"""
    # Set process priority for better real-time performance
    try:
        os.nice(-10)  # Higher priority (requires sudo)
    except Exception:
        pass  # Continue without priority boost

    # Set optimal CPU affinity for Pi 5 (use cores 0-2, leave 3 for system)
    try:
        os.sched_setaffinity(0, {0, 1, 2})
    except (AttributeError, OSError):
        pass  # Not supported on this system

    # Create and run assistant
    assistant = LumiiAssistant()
    assistant.run()

if __name__ == "__main__":
    main()
