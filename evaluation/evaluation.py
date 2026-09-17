from functools import partial

import logging

from langchain_core.messages import HumanMessage
from langsmith import Client
from langsmith.evaluation import evaluate

from state.state import SupervisorState
from agents.rag_agent import RAG

logger = logging.getLogger(__name__)

client = Client()


def ensure_dataset(dataset_name: str, eval_examples: list[dict]):
    existing = [d.name for d in client.list_datasets()]
    if dataset_name in existing:
        return client.read_dataset(dataset_name=dataset_name)

    dataset = client.create_dataset(dataset_name=dataset_name)
    for ex in eval_examples:
        client.create_example(
            inputs=ex["inputs"],
            outputs=ex["outputs"],
            dataset_id=dataset.id,
        )
    return dataset


ROUTING_EXAMPLES = [
    {
        "inputs": {"input": "How many employees do we have?"},
        "outputs": {"expected_route": "sql"},
    },
    {
        "inputs": {"input": "What did we learn about SQL security from past projects?"},
        "outputs": {"expected_route": "rag"},
    },
    {
        "inputs": {"input": "Chart our sales by region"},
        "outputs": {"expected_route": "visu"},
    },
    {
        "inputs": {"input": "hello"},
        "outputs": {"expected_route": "convo"},
    },
    {
        "inputs": {"input": "Show me the company's performance based on "},
        "outputs": {"expected_route": "clarification"},
    },
    {
        "inputs": {"input": "What's the weather in Lebanon right now?"},
        "outputs": {"expected_route": "research"},
    },
]


def routing_target_fn(graph, thread_prefix: str, inputs: dict) -> dict:
    thread_id = f"{thread_prefix}-{abs(hash(inputs['input']))}"
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": f"eval-{thread_id}",
            "permission_level": "elevated",
        }
    }

    result = graph.invoke(
        {"messages": [HumanMessage(content=inputs["input"])]},
        config=config,
    )

    interrupted = "__interrupt__" in result

    state = graph.get_state(config)
    task_history = state.values.get("task_history", [])
    current_turn = state.values.get("turn_count", 1)
    routes_taken = [r.route for r in task_history if r.turn == current_turn]

    final_response = None
    if not interrupted:
        messages = result.get("messages", [])
        if messages:
            final_response = messages[-1].content

    return {
        "routes_taken": routes_taken,
        "interrupted": interrupted,
        "final_response": final_response,
    }


def route_correctness_evaluator(run, example) -> dict:
    expected_route = example.outputs.get("expected_route")
    outputs = run.outputs or {}

    if expected_route == "clarification":
        # Clarification doesn't get recorded in task_history (a known,
        # documented gap), so correctness here means "the graph actually
        # paused for human input", not "clarification appears in routes_taken".
        correct = outputs.get("interrupted", False)
    else:
        correct = expected_route in outputs.get("routes_taken", [])

    return {"key": "correct_route_taken", "score": int(correct)}


def no_extra_specialist_calls_evaluator(run, example) -> dict:
    """
    Flags routing that technically reached the right specialist but
    took an unreasonable number of hops to get there, e.g. bouncing
    through convo multiple times first. Not a hard pass/fail, a signal
    worth watching over time as prompts/models change.
    """
    outputs = run.outputs or {}
    routes_taken = outputs.get("routes_taken", [])
    return {"key": "hop_count", "score": len(routes_taken)}


def run_routing_evaluation(dataset_name: str, graph, name_llm: str | None = None):
    ensure_dataset(dataset_name, ROUTING_EXAMPLES)
    results = evaluate(
        partial(routing_target_fn, graph, dataset_name),
        data=dataset_name,
        evaluators=[
            route_correctness_evaluator,
            no_extra_specialist_calls_evaluator,
        ],
        experiment_prefix="routing-eval",
    )
    logger.info("%s", results)
    return results


RAG_EXAMPLES = [
    {
        "inputs": {"input": "What did we learn about SQL security from past projects?"},
        "outputs": {
            "expected_status": "done",
            "answer": "Splitting SQL access by data sensitivity at the Postgres role level is more "
                      "robust than validating generated SQL in application code.",
        },
    },
    {
        "inputs": {"input": "What went wrong with state design in a past project?"},
        "outputs": {
            "expected_status": "done",
            "answer": "Cramming everything into one shared graph state made debugging difficult; "
                      "splitting shared vs local state from the start avoids this.",
        },
    },
    {
        "inputs": {"input": "What did we learn about credential handling?"},
        "outputs": {
            "expected_status": "done",
            "answer": "Credential material such as password hashes must never be returned to a "
                      "model, only an existence/boolean check should be exposed.",
        },
    },
    {
        "inputs": {"input": "What have we learned about deploying to AWS?"},
        "outputs": {"expected_status": "partial", "answer": None},
    },
    {
        "inputs": {"input": "What's our policy on mobile app development?"},
        "outputs": {"expected_status": "partial", "answer": None},
    },
]


def rag_target_fn(inputs: dict) -> dict:
    state = SupervisorState(messages=[], current_task=inputs["input"])
    result = RAG(state)
    specialist_result = result["last_result"]
    return {
        "answer": specialist_result.summary,
        "status": specialist_result.status,
        "issue": specialist_result.issue,
    }


def grounding_evaluator(run, example) -> dict:
    """
    Checks the structural status, not the wording. This is what
    actually catches hallucination on out-of-corpus questions: an LLM
    judge comparing prose alone can be fooled by a plausible-sounding
    wrong answer, the status field can't be.
    """
    expected_status = example.outputs.get("expected_status")
    actual_status = (run.outputs or {}).get("status")
    return {"key": "grounding_status_correct", "score": int(actual_status == expected_status)}


def rag_answer_correctness_evaluator(llm, run, example) -> dict:
    reference = example.outputs.get("answer")
    if reference is None:
        return {
            "key": "answer_correctness",
            "score": None,
            "comment": "skipped — no reference answer (out-of-corpus case, graded by grounding_evaluator instead)",
        }

    prediction = (run.outputs or {}).get("answer", "")
    question = example.inputs.get("input", "")

    grading_prompt = (
        f"Question: {question}\n"
        f"Reference answer: {reference}\n"
        f"Model answer: {prediction}\n\n"
        "Does the model answer correctly address the question and align with the "
        "reference answer? Reply with only 'correct' or 'incorrect'."
    )
    verdict = llm.invoke(grading_prompt).content.strip().lower()
    score = 1 if "correct" in verdict and "incorrect" not in verdict else 0
    return {"key": "answer_correctness", "score": score}


def run_rag_evaluation(dataset_name: str, name_llm: str | None = None):
    from services.llm import llm  # local import to avoid a hard dependency at module load
    model = llm(name_llm) if name_llm else llm()
    ensure_dataset(dataset_name, RAG_EXAMPLES)
    results = evaluate(
        rag_target_fn,
        data=dataset_name,
        evaluators=[
            grounding_evaluator,
            partial(rag_answer_correctness_evaluator, model),
        ],
        experiment_prefix="rag-eval",
    )
    logger.info("%s", results)
    return results