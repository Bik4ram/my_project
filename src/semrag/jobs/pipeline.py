"""
Pipeline stages and progress calculation for indexing jobs.
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Dict, List


class PipelineStage(Enum):
    """Stages of the indexing pipeline with their relative weights."""
    QUEUED = auto()  # 0% - waiting to start
    SCANNING = auto()  # 5% - scanning files
    PARSING = auto()  # 20% - parsing files
    IR = auto()  # 45% - building IR
    CHUNKING = auto()  # 65% - chunking
    EMBEDDING = auto()  # 90% - embedding chunks
    PERSISTING = auto()  # 100% - persisting to database
    COMPLETED = auto()  # 100% - job done
    FAILED = auto()  # 100% - job failed (but we don't progress further)
    CANCELLED = auto()  # 100% - job cancelled

    def weight(self) -> float:
        """Return the weight of this stage (0.0 to 1.0)."""
        weights = {
            PipelineStage.QUEUED: 0.0,
            PipelineStage.SCANNING: 0.05,
            PipelineStage.PARSING: 0.20,
            PipelineStage.IR: 0.45,
            PipelineStage.CHUNKING: 0.65,
            PipelineStage.EMBEDDING: 0.90,
            PipelineStage.PERSISTING: 1.0,
            PipelineStage.COMPLETED: 1.0,
            PipelineStage.FAILED: 1.0,
            PipelineStage.CANCELLED: 1.0,
        }
        return weights[self]


class ProgressCalculator:
    """Calculate overall progress based on completed stages and current stage progress."""

    def __init__(self, stage_weights: Dict[PipelineStage, float] = None):
        if stage_weights is None:
            # Build weights from the PipelineStage enum
            self.stage_weights = {stage: stage.weight() for stage in PipelineStage}
        else:
            self.stage_weights = stage_weights

    def calculate_progress(
        self,
        completed_stages: List[PipelineStage],
        current_stage: PipelineStage,
        current_stage_progress: float = 0.0,
    ) -> int:
        """
        Calculate progress percentage.

        :param completed_stages: List of stages that have been fully completed.
        :param current_stage: The stage currently being processed.
        :param current_stage_progress: Progress within the current stage (0.0 to 1.0).
        :return: Overall progress percentage (0 to 100).
        """
        # Start with the weight of the most recent completed stage
        progress = 0.0
        if completed_stages:
            # Get the highest weight among completed stages
            progress = max(self.stage_weights[stage] for stage in completed_stages)

        # Add the progress within the current stage
        if current_stage in self.stage_weights:
            stage_weight = self.stage_weights[current_stage]
            # The weight of the current stage contributes from its start weight to its end weight
            # We assume that the stage starts at the weight of the previous completed stage
            # For simplicity, we'll use the stage's weight as the end of the stage and
            # the start of the stage as the weight of the previous stage.
            # But note: our stages are not necessarily contiguous in weight? They are increasing.
            # We'll calculate the increment as (current_stage_progress * (stage_weight - start_weight))
            # where start_weight is the weight of the previous stage (or 0 if none).
            # However, we don't have the previous stage. We'll approximate by:
            #   progress = weight of the most recent completed stage
            #   then add current_stage_progress * (stage_weight - weight of most recent completed stage)
            # But note: the most recent completed stage might not be the immediate predecessor.
            # We'll instead use the stage's weight as the maximum for that stage and
            # assume that the stage's progress is linear from 0 to stage_weight.
            # Actually, we want the overall progress to be the weight of the completed stages
            # plus the fraction of the current stage multiplied by the stage's weight range.
            # Let's redefine: each stage has a start and end weight.
            # We'll define the start weight of a stage as the end weight of the previous stage.
            # We'll hardcode the start and end weights for simplicity.

            # Alternatively, we can think of the pipeline as a sequence of stages with weights:
            #   QUEUED: [0.0, 0.0]
            #   SCANNING: [0.0, 0.05]
            #   PARSING: [0.05, 0.25]
            #   IR: [0.25, 0.70]
            #   CHUNKING: [0.70, 0.80]
            #   EMBEDDING: [0.80, 0.95]
            #   PERSISTING: [0.95, 1.0]
            # But note: our weights above are the end weights.

            # Let's define the start and end weights for each stage:
            stage_ranges = {
                PipelineStage.QUEUED: (0.0, 0.0),
                PipelineStage.SCANNING: (0.0, 0.05),
                PipelineStage.PARSING: (0.05, 0.25),
                PipelineStage.IR: (0.25, 0.70),
                PipelineStage.CHUNKING: (0.70, 0.80),
                PipelineStage.EMBEDDING: (0.80, 0.95),
                PipelineStage.PERSISTING: (0.95, 1.0),
                PipelineStage.COMPLETED: (1.0, 1.0),
                PipelineStage.FAILED: (1.0, 1.0),
                PipelineStage.CANCELLED: (1.0, 1.0),
            }

            start_weight, end_weight = stage_ranges[current_stage]
            progress = start_weight + (current_stage_progress * (end_weight - start_weight))

            # But note: we have already accounted for completed stages? Actually, if we are in a stage,
            # the previous stages are completed. So we should set progress to the end weight of the last
            # completed stage plus the progress in the current stage.
            # However, the above formula already does that if we set start_weight to the end of the previous stage.
            # And we have defined the ranges such that the start of a stage is the end of the previous stage.

            # However, we are not passing the completed stages to this function? We are passing completed_stages.
            # We can ignore completed_stages and just use the current stage and its progress?
            # But note: the current stage might be, say, PARSING, and we have completed QUEUED and SCANNING.
            # The start_weight for PARSING is 0.05 (which is the end of SCANNING) and end_weight is 0.25.
            # So if current_stage_progress is 0.5, then progress = 0.05 + 0.5*(0.25-0.05) = 0.05 + 0.1 = 0.15.
            # This is correct.

            # However, what if we have skipped a stage? We shouldn't skip stages.

            # We'll use the stage_ranges to compute progress based solely on the current stage and its progress.
            # But note: the QUEUED stage has start and end 0.0, so progress will be 0.0 until we leave QUEUED.
            # We'll treat QUEUED as a special case? Actually, we want QUEUED to be 0% and then when we start
            # SCANNING, we jump to 0% and then progress to 5%.

            # We'll change the approach: we'll compute progress as the weight of the most recent completed stage
            # plus the progress in the current stage (if any) weighted by the stage's weight range.

            # Let's revert to the original plan and use the completed_stages to get the base progress.

            # We'll do:
            #   base_progress = max(self.stage_weights[stage] for stage in completed_stages) if completed_stages else 0.0
            #   Then, if current_stage is not in completed_stages, we add the progress in the current stage.
            #   The current stage's weight range is from base_progress to self.stage_weights[current_stage]?
            #   But note: the stage's weight might be less than base_progress if we have completed a later stage?
            #   That shouldn't happen.

            # We'll do it this way:

            if completed_stages:
                base_progress = max(self.stage_weights[stage] for stage in completed_stages)
            else:
                base_progress = 0.0

            # If the current stage is already completed, then we don't add anything.
            if current_stage in completed_stages:
                progress = base_progress
            else:
                # The current stage's weight is the end of the stage.
                stage_end = self.stage_weights[current_stage]
                # The progress within the current stage is from base_progress to stage_end.
                progress = base_progress + (current_stage_progress * (stage_end - base_progress))

        return int(progress * 100)