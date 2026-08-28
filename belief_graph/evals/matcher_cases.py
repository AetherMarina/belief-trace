from dataclasses import dataclass

from belief_graph.core import ExtractedTriplet
from belief_graph.matching import MatchType


@dataclass
class MatcherCase:
    name: str
    existing: ExtractedTriplet
    new: ExtractedTriplet
    expected: MatchType
    similarity_score: float


DEV_CASES = [
    MatcherCase(
        name="exact_same",
        existing=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Vulnerable",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Vulnerable",
        ),
        expected=MatchType.SAME,
        similarity_score=1.0,
    ),

    MatcherCase(
        name="same_paraphrase",
        existing=ExtractedTriplet(
            subject="Others",
            relation="ARE",
            object="Untrustworthy",
        ),
        new=ExtractedTriplet(
            subject="Others",
            relation="CANNOT",
            object="Be Trusted",
        ),
        expected=MatchType.SAME,
        similarity_score=0.93,
    ),

    MatcherCase(
        name="related_but_different",
        existing=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Vulnerable",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="FEELS",
            object="Unsafe",
        ),
        expected=MatchType.DIFFERENT,
        similarity_score=0.91,
    ),

    MatcherCase(
        name="clearly_different",
        existing=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Curious",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Vulnerable",
        ),
        expected=MatchType.DIFFERENT,
        similarity_score=0.72,
    ),

    MatcherCase(
        name="direct_contradiction",
        existing=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Safe",
        ),
        new=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Dangerous",
        ),
        expected=MatchType.CONTRADICTS,
        similarity_score=0.88,
    ),

    MatcherCase(
        name="belief_revision",
        existing=ExtractedTriplet(
            subject="Others",
            relation="ARE",
            object="Inherently Evil",
        ),
        new=ExtractedTriplet(
            subject="Others",
            relation="INCLUDE",
            object="Good and Bad People",
        ),
        expected=MatchType.CONTRADICTS,
        similarity_score=0.86,
    ),

    MatcherCase(
        name="related_concepts_not_same",
        existing=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Unpredictable",
        ),
        new=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Dangerous",
        ),
        expected=MatchType.DIFFERENT,
        similarity_score=0.84,
    ),
    MatcherCase(
        name="direct_negation_contradiction",
        existing=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Capable",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="IS NOT",
            object="Capable",
        ),
        expected=MatchType.CONTRADICTS,
        similarity_score=0.98,
    ),

    MatcherCase(
        name="negated_paraphrase_same",
        existing=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Incapable",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="CANNOT",
            object="Do It",
        ),
        expected=MatchType.SAME,
        similarity_score=0.94,
    ),
]

HELD_OUT_CASES = [
    MatcherCase(
        name="same_worthless_paraphrase",
        existing=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Worthless",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="HAS",
            object="No Worth",
        ),
        expected=MatchType.SAME,
        similarity_score=0.95,
    ),

    MatcherCase(
        name="same_world_unfair_paraphrase",
        existing=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Unfair",
        ),
        new=ExtractedTriplet(
            subject="World",
            relation="LACKS",
            object="Fairness",
        ),
        expected=MatchType.SAME,
        similarity_score=0.94,
    ),

    MatcherCase(
        name="same_others_dishonest_paraphrase",
        existing=ExtractedTriplet(
            subject="Others",
            relation="ARE",
            object="Dishonest",
        ),
        new=ExtractedTriplet(
            subject="Others",
            relation="DO NOT",
            object="Tell the Truth",
        ),
        expected=MatchType.SAME,
        similarity_score=0.92,
    ),

    MatcherCase(
        name="same_rejection_paraphrase",
        existing=ExtractedTriplet(
            subject="Others",
            relation="WILL",
            object="Reject Me",
        ),
        new=ExtractedTriplet(
            subject="Others",
            relation="ARE BOUND TO",
            object="Reject Me",
        ),
        expected=MatchType.SAME,
        similarity_score=0.96,
    ),

    MatcherCase(
        name="same_chaotic_paraphrase",
        existing=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Chaotic",
        ),
        new=ExtractedTriplet(
            subject="World",
            relation="LACKS",
            object="Order",
        ),
        expected=MatchType.SAME,
        similarity_score=0.91,
    ),

    # Slightly harder SAME case
    MatcherCase(
        name="same_helpless_paraphrase",
        existing=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Helpless",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="CANNOT",
            object="Influence Outcomes",
        ),
        expected=MatchType.SAME,
        similarity_score=0.89,
    ),


    # ============================================================
    # CONTRADICTS
    # ============================================================

    MatcherCase(
        name="contradiction_worthy_worthless",
        existing=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Worthy",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Worthless",
        ),
        expected=MatchType.CONTRADICTS,
        similarity_score=0.94,
    ),

    MatcherCase(
        name="contradiction_trust",
        existing=ExtractedTriplet(
            subject="Others",
            relation="CAN",
            object="Be Trusted",
        ),
        new=ExtractedTriplet(
            subject="Others",
            relation="CANNOT",
            object="Be Trusted",
        ),
        expected=MatchType.CONTRADICTS,
        similarity_score=0.98,
    ),

    MatcherCase(
        name="contradiction_predictability",
        existing=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Predictable",
        ),
        new=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Unpredictable",
        ),
        expected=MatchType.CONTRADICTS,
        similarity_score=0.97,
    ),

    MatcherCase(
        name="contradiction_deserving_love",
        existing=ExtractedTriplet(
            subject="Self",
            relation="DESERVES",
            object="Love",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="DOES NOT DESERVE",
            object="Love",
        ),
        expected=MatchType.CONTRADICTS,
        similarity_score=0.98,
    ),

    # Tests qualification of an absolute belief
    MatcherCase(
        name="contradiction_abandonment_exception",
        existing=ExtractedTriplet(
            subject="Others",
            relation="ALWAYS",
            object="Abandon Me",
        ),
        new=ExtractedTriplet(
            subject="Others",
            relation="SOMETIMES",
            object="Stay With Me",
        ),
        expected=MatchType.CONTRADICTS,
        similarity_score=0.87,
    ),

    # Another generalized-belief revision
    MatcherCase(
        name="contradiction_absolute_safety",
        existing=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Completely Unsafe",
        ),
        new=ExtractedTriplet(
            subject="World",
            relation="CAN BE",
            object="Safe",
        ),
        expected=MatchType.CONTRADICTS,
        similarity_score=0.86,
    ),


    # ============================================================
    # BROAD SAME / ACCEPTABLE MERGE
    # ============================================================

    MatcherCase(
        name="same_worthless_powerless",
        existing=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Worthless",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Powerless",
        ),
        expected=MatchType.SAME,
        similarity_score=0.88,
    ),

    # ============================================================
    # DIFFERENT
    # ============================================================

    MatcherCase(
        name="different_unfair_unpredictable",
        existing=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Unfair",
        ),
        new=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Unpredictable",
        ),
        expected=MatchType.DIFFERENT,
        similarity_score=0.82,
    ),

    MatcherCase(
        name="different_critical_unreliable",
        existing=ExtractedTriplet(
            subject="Others",
            relation="ARE",
            object="Critical",
        ),
        new=ExtractedTriplet(
            subject="Others",
            relation="ARE",
            object="Unreliable",
        ),
        expected=MatchType.DIFFERENT,
        similarity_score=0.81,
    ),

    MatcherCase(
        name="different_lonely_incompetent",
        existing=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Lonely",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Incompetent",
        ),
        expected=MatchType.DIFFERENT,
        similarity_score=0.74,
    ),

    MatcherCase(
        name="different_dangerous_demanding",
        existing=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Dangerous",
        ),
        new=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Demanding",
        ),
        expected=MatchType.DIFFERENT,
        similarity_score=0.78,
    ),

    # Hard boundary: emotional state vs core self-evaluation
    MatcherCase(
        name="different_feeling_vs_identity",
        existing=ExtractedTriplet(
            subject="Self",
            relation="FEELS",
            object="Ashamed",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Defective",
        ),
        expected=MatchType.DIFFERENT,
        similarity_score=0.90,
    ),
]

CALIBRATION_CASES = [
    # SAME / acceptable recurrence
    MatcherCase(
        name="same_unlovable",
        existing=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Unlovable",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="CANNOT",
            object="Be Loved",
        ),
        expected=MatchType.SAME,
        similarity_score=0.94,
    ),

    MatcherCase(
        name="same_no_value",
        existing=ExtractedTriplet(
            subject="Self",
            relation="HAS",
            object="No Value",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Worthless",
        ),
        expected=MatchType.SAME,
        similarity_score=0.95,
    ),

    MatcherCase(
        name="same_unreliable",
        existing=ExtractedTriplet(
            subject="Others",
            relation="ARE",
            object="Unreliable",
        ),
        new=ExtractedTriplet(
            subject="Others",
            relation="CANNOT",
            object="Be Relied Upon",
        ),
        expected=MatchType.SAME,
        similarity_score=0.92,
    ),

    MatcherCase(
        name="same_no_control",
        existing=ExtractedTriplet(
            subject="Self",
            relation="HAS",
            object="No Control",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="CANNOT",
            object="Control Outcomes",
        ),
        expected=MatchType.SAME,
        similarity_score=0.93,
    ),

    MatcherCase(
        name="same_world_hostile",
        existing=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Hostile",
        ),
        new=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Adversarial",
        ),
        expected=MatchType.SAME,
        similarity_score=0.90,
    ),

    MatcherCase(
        name="same_rejection",
        existing=ExtractedTriplet(
            subject="Others",
            relation="WILL",
            object="Reject Me",
        ),
        new=ExtractedTriplet(
            subject="Others",
            relation="DO NOT",
            object="Accept Me",
        ),
        expected=MatchType.SAME,
        similarity_score=0.91,
    ),

    MatcherCase(
        name="same_incompetent",
        existing=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Incompetent",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="LACKS",
            object="Competence",
        ),
        expected=MatchType.SAME,
        similarity_score=0.95,
    ),

    MatcherCase(
        name="same_world_unjust",
        existing=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Unjust",
        ),
        new=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Unfair",
        ),
        expected=MatchType.SAME,
        similarity_score=0.92,
    ),

    MatcherCase(
        name="same_unlovable_unwanted",
        existing=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Unlovable",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Unwanted",
        ),
        expected=MatchType.SAME,
        similarity_score=0.91,
    ),

    MatcherCase(
        name="different_helpless_incompetent",
        existing=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Helpless",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Incompetent",
        ),
        expected=MatchType.DIFFERENT,
        similarity_score=0.89,
    ),

    MatcherCase(
        name="different_dangerous_unpredictable",
        existing=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Dangerous",
        ),
        new=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Unpredictable",
        ),
        expected=MatchType.DIFFERENT,
        similarity_score=0.87,
    ),

    MatcherCase(
        name="different_hostile_unfair",
        existing=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Hostile",
        ),
        new=ExtractedTriplet(
            subject="World",
            relation="IS",
            object="Unfair",
        ),
        expected=MatchType.DIFFERENT,
        similarity_score=0.85,
    ),

    MatcherCase(
        name="different_critical_untrustworthy",
        existing=ExtractedTriplet(
            subject="Others",
            relation="ARE",
            object="Critical",
        ),
        new=ExtractedTriplet(
            subject="Others",
            relation="ARE",
            object="Untrustworthy",
        ),
        expected=MatchType.DIFFERENT,
        similarity_score=0.84,
    ),

    MatcherCase(
        name="same_abandon_reject",
        existing=ExtractedTriplet(
            subject="Others",
            relation="WILL",
            object="Abandon Me",
        ),
        new=ExtractedTriplet(
            subject="Others",
            relation="WILL",
            object="Reject Me",
        ),
        expected=MatchType.SAME,
        similarity_score=0.92,
    ),

    MatcherCase(
        name="different_defective_ashamed",
        existing=ExtractedTriplet(
            subject="Self",
            relation="IS",
            object="Defective",
        ),
        new=ExtractedTriplet(
            subject="Self",
            relation="FEELS",
            object="Ashamed",
        ),
        expected=MatchType.DIFFERENT,
        similarity_score=0.90,
    ),
]
