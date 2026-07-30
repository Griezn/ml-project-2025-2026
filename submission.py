"""
Final submission file for Knights Archers Zombies tournament.
Integrates IPPO agent policy and YOLO visual detector pipeline.
"""

from ippo_agent import (
    CustomWrapper,
    CustomPredictFunction,
    CustomZombieDetectorFunction,
)

__all__ = ["CustomWrapper", "CustomPredictFunction", "CustomZombieDetectorFunction"]
