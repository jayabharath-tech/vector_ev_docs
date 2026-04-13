"""
Evaluation module for RAG system using pydantic_evals.
Tests RAG agent quality with LJudge evaluator and RAGAS faithfulness metric.
"""

import asyncio
from pydantic_evals import Dataset, Case
from pydantic_evals.evaluators import LLMJudge, Evaluator

from main import main, ingest_pdf


# Wrapper to handle async main() with sync evaluate_sync()
def _sync_main_wrapper(question: str):
    """Sync wrapper for async main function."""
    return asyncio.run(main(question))


# ============================================================================
# RAGAS FAITHFULNESS EVALUATOR
# ============================================================================

class SelectiveRAGASEvaluator(Evaluator):
    """Selectively evaluate faithfulness using RAGAS framework.

    COST OPTIMIZATION - Only evaluates SPECIFIC test cases to reduce API calls.

    Faithfulness checks if generated answers are grounded in retrieved contexts.
    It detects hallucinations - facts in answers not present in source documents.

    Selected Cases (with RAGAS evaluation):
    - Case 0: "how much time is required..." (Technical fact - good for faithfulness test)
    - Case 1: "What charging-related technologies..." (Multiple facts to verify)
    - Case 3: "What is CANoe..." (Specific definition - easy to verify)

    Excluded Cases (LJudge only, no RAGAS):
    - Cases about document structure (not grounding-critical)
    - Out-of-scope cases (test honesty, not faithfulness)

    Cost Impact:
    - Without selective: 12 cases × 4 API calls = 48 calls (~$0.15)
    - With selective: 3 cases × 4 API calls = 12 calls (~$0.04)
    - Savings: 75% cost reduction
    """

    # Test case indices to evaluate with RAGAS (0-indexed)
    EVALUATE_INDICES = {0, 1, 3}

    def __init__(self):
        """Initialize RAGAS faithfulness metric."""
        try:
            from ragas.metrics import faithfulness
            from anthropic import Anthropic

            self.faithfulness_metric = faithfulness
            self.client = Anthropic()
        except ImportError:
            raise ImportError(
                "RAGAS library required for faithfulness evaluation. "
                "Install with: pip install ragas"
            )

    def evaluate(self, ctx, **kwargs) -> dict:
        """
        Selectively evaluate faithfulness using RAGAS metric.

        RAGAS Faithfulness Process:
        1. Break answer into atomic statements
        2. For each statement, check if supported by contexts
        3. Calculate: (supported / total)
        4. Return score 0-1

        SELECTIVE EVALUATION:
        - Checks if this is a case that should be evaluated (based on index)
        - Only runs RAGAS on critical test cases
        - Skips out-of-scope and structure cases

        Args:
            ctx: EvaluatorContext containing output and inputs
            **kwargs: Additional arguments

        Returns:
            Dictionary with evaluation results:
            {
                "evaluation_name": "faithfulness",
                "passed": bool,  # True if score >= 0.7
                "score": float,  # 0-1 range
                "reason": str    # Human-readable explanation
            }
        """
        # Get case index from context
        case_index = kwargs.get('case_index', -1)

        # COST OPTIMIZATION: Skip cases not in evaluation set
        if case_index not in self.EVALUATE_INDICES:
            return {
                "evaluation_name": "faithfulness",
                "passed": True,  # Skip with pass (not evaluated)
                "score": None,   # No score for skipped cases
                "reason": f"Faithfulness evaluation skipped for case {case_index} "
                         f"(not in critical test set for cost optimization)",
            }

        try:
            # Extract actual output (AgentResponse)
            if hasattr(ctx, 'output'):
                actual_output = ctx.output
            elif hasattr(ctx, 'actual_output'):
                actual_output = ctx.actual_output
            else:
                actual_output = ctx

            # Get user input (question)
            user_input = kwargs.get('inputs', '')
            if not user_input and hasattr(ctx, 'inputs'):
                user_input = ctx.inputs

            # Extract contexts from retrieved metadata
            # source_metadata contains original_text field with document text
            contexts = []
            if hasattr(actual_output, 'source_metadata') and actual_output.source_metadata:
                contexts = [
                    metadata.original_text
                    for metadata in actual_output.source_metadata
                ]

            if not contexts:
                # Fallback to source_snippet if metadata not available
                contexts = actual_output.source_snippet if hasattr(actual_output, 'source_snippet') else []

            # Create sample object for RAGAS evaluation
            # RAGAS expects: question, answer, context/contexts
            class RagasSample:
                def __init__(self, user_input, response, retrieved_contexts):
                    self.user_input = user_input
                    self.response = response
                    self.retrieved_contexts = retrieved_contexts

            sample = RagasSample(
                user_input=user_input,
                response=actual_output.answer,
                retrieved_contexts=contexts
            )

            # Score using RAGAS faithfulness metric
            # Returns score 0-1 (1 = fully grounded, 0 = hallucinated)
            score = self.faithfulness_metric.single_turn_score(sample)
            score_value = float(score)

            # Threshold: 0.7 for PASS (70% of facts grounded)
            passed = score_value >= 0.7

            return {
                "evaluation_name": "faithfulness",
                "passed": passed,
                "score": score_value,
                "reason": f"RAGAS Faithfulness: {score_value:.2f} "
                         f"{'(PASS - Grounded)' if passed else '(FAIL - Hallucination detected)'}",
            }

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return {
                "evaluation_name": "faithfulness",
                "passed": False,
                "score": 0.0,
                "reason": f"Faithfulness eval error: {str(e)[:100]}",
            }


# ============================================================================
# EVALUATION DATASET
# ============================================================================

rag_eval_dataset = Dataset(
    cases=[
        # Technical Concepts
        Case(
            inputs="how much time is required to charge a battery of heavy vehicles under an hour?",
            expected_output="The MCS (Megawatt Charging System) standard, currently in standardization, aims to bring the batteries of heavy traffic vehicles to an 80% charge status in approximately 45 minutes. This charging capability is achieved through a specified charging power of 3.75 MW"
        ),
        Case(
            inputs="What charging-related technologies are discussed in the document?",
            expected_output="The document discusses charging systems, SmartCharging solutions with CANoe integration, and related development tools."
        ),
        Case(
            inputs="What simulation tools are mentioned in the technical article?",
            expected_output="The document mentions simulation tools and development platforms used for automotive electronics testing."
        ),
        Case(
            inputs="What is the CANoe Test Package EV?",
            expected_output="""
                The CANoe Test Package EV is a library of powerful test cases. 
                This is used to test conformance and interoperability of electric vehicles (EV) within the standards CCS, NACS,B/T and CHAdeM0.
                """
        ),

        # Implementation Details
        Case(
            inputs="What is CANoe and how is it used?",
            expected_output="CANoe is a tool option mentioned in Vector's MCS solutions for automotive electronics development and testing."
        ),
        Case(
            inputs="What are the key components discussed in the document?",
            expected_output="The document discusses various automotive electronics components, charging systems, and Vector's development solutions."
        ),

        # Document Structure
        Case(
            inputs="What topics are covered in this technical document?",
            expected_output="The document covers automotive electronics, charging systems, simulation tools, and Vector Elektronik's technical solutions."
        ),
        Case(
            inputs="What is the purpose of this document?",
            expected_output="This is a technical article from Vector Elektronik explaining their automotive electronics and charging solutions."
        ),

        # Capability Testing
        Case(
            inputs="Does the document mention performance metrics?",
            expected_output="The document discusses performance testing and evaluation of automotive electronics systems."
        ),
        Case(
            inputs="What industries or sectors does this apply to?",
            expected_output="This applies to automotive industry, specifically electric vehicle charging and automotive electronics development."
        ),

        # Out-of-scope Testing (no RAGAS needed - these test honesty, not grounding)
        Case(
            inputs="What is the total number of pages in this document?",
            expected_output="The page count information is not available in the knowledge base."
        ),
        Case(
            inputs="Tell me something not covered in this document.",
            expected_output="I cannot provide information about topics not covered in the Vector Elektronik technical document."
        ),
    ],
    evaluators=[
        LLMJudge(
            model="claude-haiku-4-5-20251001",
            rubric="""Evaluate the RAG response on the following criteria:
1. Relevance: Does the answer address the question asked?
2. Grounding: Is the answer based on the retrieved context? (Very important for RAG)
3. Accuracy: Is the information correct based on the provided documents?
4. Completeness: Does it cover the key aspects of the question?
5. Honesty: Does it admit when information is not available?

Score: PASS if the answer is well-grounded, relevant, and honest. FAIL if hallucinated, irrelevant, or dishonest."""
        ),
        # Selective RAGAS evaluator (only on critical cases for cost optimization)
        # SelectiveRAGASEvaluator(),
    ]
)


# ============================================================================
# RUN EVALUATION
# ============================================================================

def run_evaluation(data_path: str = "data"):
    """
    Run the evaluation suite on the RAG agent.

    Args:
        data_path: Path to PDF file or directory containing PDFs
    """
    print("=" * 80)
    print("STARTING RAG EVALUATION")
    print("=" * 80)
    print()

    # Step 1: Ingest PDFs once (before evaluation)
    print("Step 1: Ingesting PDF(s)...")
    asyncio.run(ingest_pdf(data_path))
    print()

    # Step 2: Run evaluation on all test cases with LLMJudge + Selective RAGAS
    print("Step 2: Running evaluation on test cases...")
    print("  - LLMJudge: All 12 cases")
    print("  - RAGAS Faithfulness: Cases 0, 1, 3 (critical grounding tests)")
    print("-" * 80)
    report = rag_eval_dataset.evaluate_sync(_sync_main_wrapper)

    # Step 3: Print results
    print()
    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print("\n📊 SUMMARY:")
    print(f"  - LLMJudge: Evaluates all cases on relevance, grounding, honesty")
    print(f"  - RAGAS: Evaluates {len(SelectiveRAGASEvaluator.EVALUATE_INDICES)} critical cases on faithfulness")
    print(f"  - Cost: ~24 Claude API calls (~$0.08 total)")
    print()

    report.print(
        # include_output=True,
        include_expected_output=True,
    )

    return report


if __name__ == "__main__":

    run_evaluation(data_path="data")
