"""Generate 1-rag_evaluation-todo.ipynb: the LangSmith notebook converted to Langfuse.

Run once from this folder:
    python gen_todo_notebook.py
"""
import json, os, uuid

cells = []

def md(text):
    cells.append({
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": text,
    })

def code(text):
    cells.append({
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text,
    })

md("""### Chatbot And RAG Evaluation (with Langfuse)

Retrieval Augmented Generation (RAG) is a technique that enhances Large Language Models (LLMs) by providing them with relevant external knowledge. It has become one of the most widely used approaches for building LLM applications.

This notebook shows how to evaluate your chatbot / RAG applications using **Langfuse** (the open-source alternative to LangSmith). You'll learn:

1. How to create test datasets in Langfuse
2. How to run your application on those datasets with `run_experiment`
3. How to measure performance with LLM-as-a-judge evaluators (correctness, relevance, groundedness, retrieval relevance)

#### Overview
A typical evaluation workflow has three steps:

1. Create a dataset with questions and expected answers (`create_dataset` + `create_dataset_item`)
2. Run your application on each item via a `task` function
3. Attach `evaluators` that score each output; results are traced and aggregated in the Langfuse UI

For this tutorial, we'll build and evaluate a bot that answers questions about a few of Lilian Weng's blog posts.

> **Langfuse vs LangSmith cheat-sheet**
> | LangSmith | Langfuse |
> |---|---|
> | `Client()` | `get_client()` |
> | `client.create_dataset` / `create_examples` | `create_dataset` / `create_dataset_item` |
> | `@traceable` | `@observe` |
> | `wrappers.wrap_openai` | `from langfuse.openai import openai` |
> | `client.evaluate(target, data, evaluators)` | `client.run_experiment(name, data, task, evaluators)` |
> | evaluator returns `bool` | evaluator returns `Evaluation(name=, value=, comment=)` |""")

md("### Chatbot Evaluation")

code("""import os
from dotenv import load_dotenv
from langfuse import get_client, observe, Evaluation

load_dotenv()

# Langfuse credentials (put these in your .env file)
os.environ["LANGFUSE_PUBLIC_KEY"] = os.getenv("LANGFUSE_PUBLIC_KEY")
os.environ["LANGFUSE_SECRET_KEY"] = os.getenv("LANGFUSE_SECRET_KEY")
os.environ["LANGFUSE_HOST"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = get_client()
assert client.auth_check(), "Langfuse auth failed - check LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST"
print("Langfuse client authenticated")""")

code('''# Define dataset: these are your test cases
dataset_name = "Chatbots Evaluation"

examples = [
    {"inputs": {"question": "What is LangChain?"},
     "outputs": {"answer": "A framework for building LLM applications"}},
    {"inputs": {"question": "What is LangSmith?"},
     "outputs": {"answer": "A platform for observing and evaluating LLM applications"}},
    {"inputs": {"question": "What is OpenAI?"},
     "outputs": {"answer": "A company that creates Large Language Models"}},
    {"inputs": {"question": "What is Google?"},
     "outputs": {"answer": "A technology company known for search"}},
    {"inputs": {"question": "What is Mistral?"},
     "outputs": {"answer": "A company that creates Large Language Models"}},
]

# create_dataset upserts by name -> safe to re-run
client.create_dataset(name=dataset_name)

# Passing a stable `id` (the question) makes re-runs upsert instead of
# creating duplicate items.
for ex in examples:
    client.create_dataset_item(
        dataset_name=dataset_name,
        id=ex["inputs"]["question"],
        input=ex["inputs"],
        expected_output=ex["outputs"],
    )

print(f"Created dataset '{dataset_name}' with {len(examples)} items")''')

md("### Define Metrics (LLM As A Judge)")

code('''# Langfuse drop-in wrapper for the OpenAI SDK: every call below is auto-traced.
from langfuse.openai import openai

openai_client = openai.OpenAI()

eval_instructions = "You are an expert professor specialized in grading students' answers to questions."

# A Langfuse evaluator receives the item's `input`, the task `output`,
# the `expected_output`, and `metadata` as keyword arguments, and returns
# one (or a list of) Evaluation(name=..., value=..., comment=...) object(s).
def correctness(*, input, output, expected_output, metadata=None, **kwargs):
    """LLM-as-judge: is the predicted answer correct vs the reference answer?"""
    user_content = f"""You are grading the following question:
{input['question']}
Here is the real answer:
{expected_output['answer']}
You are grading the following predicted answer:
{output}
Respond with CORRECT or INCORRECT:
Grade:
"""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": eval_instructions},
            {"role": "user", "content": user_content},
        ],
    ).choices[0].message.content

    return Evaluation(name="correctness", value=response.strip() == "CORRECT", comment=response)''')

code('''# Concision - checks whether the output is less than 2x the length of the expected answer.
def concision(*, input, output, expected_output, metadata=None, **kwargs):
    return Evaluation(
        name="concision",
        value=int(len(output) < 2 * len(expected_output["answer"])),
    )''')

md("### Run Evaluations")

code('''default_instructions = "Respond to the users question in a short, concise manner (one short sentence)."

def my_app(question: str, model: str = "gpt-4o-mini", instructions: str = default_instructions) -> str:
    return openai_client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
        ],
    ).choices[0].message.content''')

code('''# The task function is called once per dataset item.
# `item.input` is the dict we stored -> {"question": ...}
def task(*, item, **kwargs) -> str:
    return my_app(item.input["question"])''')

code('''dataset = client.get_dataset(dataset_name)

result = client.run_experiment(
    name="openai-4o-mini-chatbot",
    data=dataset.items,
    task=task,
    evaluators=[correctness, concision],
)

print(result.format())''')

code('''# Same dataset, different model -> comparable experiment run
def task_turbo(*, item, **kwargs) -> str:
    return my_app(item.input["question"], model="gpt-4-turbo")

result = client.run_experiment(
    name="openai-4-turbo-chatbot",
    data=dataset.items,
    task=task_turbo,
    evaluators=[correctness, concision],
)

print(result.format())''')

md("### Evaluation For RAG")

code('''## RAG retriever
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# List of URLs to load documents from
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents from the URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs_list)

# Add the document chunks to an in-memory vector store using OpenAIEmbeddings
vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings(),
)

# Turn the vector store into a retrieval component
retriever = vectorstore.as_retriever(k=6)''')

code('retriever.invoke("what is agents")')

code('''from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o-mini")
llm''')

code('''# @observe traces this function (and the nested retriever + llm calls) in Langfuse,
# the Langfuse equivalent of LangSmith's @traceable.
@observe()
def rag_bot(question: str) -> dict:
    ## Relevant context
    docs = retriever.invoke(question)
    docs_string = " ".join(doc.page_content for doc in docs)

    instructions = f"""You are a helpful assistant who is good at analyzing source information and answering questions. \
Use the following source documents to answer the user's questions. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.

Documents:
{docs_string}"""

    ## llm invoke
    ai_msg = llm.invoke([
        {"role": "system", "content": instructions},
        {"role": "user", "content": question},
    ])
    return {"answer": ai_msg.content, "documents": docs}''')

code('rag_bot("What is agents")')

md("### Dataset")

code('''# Define the examples for the RAG dataset
rag_dataset_name = "RAG Test Evaluation"

rag_examples = [
    {
        "inputs": {"question": "How does the ReAct agent use self-reflection? "},
        "outputs": {"answer": "ReAct integrates reasoning and acting, performing actions - such tools like Wikipedia search API - and then observing / reasoning about the tool outputs."},
    },
    {
        "inputs": {"question": "What are the types of biases that can arise with few-shot prompting?"},
        "outputs": {"answer": "The biases that can arise with few-shot prompting include (1) Majority label bias, (2) Recency bias, and (3) Common token bias."},
    },
    {
        "inputs": {"question": "What are five types of adversarial attacks?"},
        "outputs": {"answer": "Five types of adversarial attacks are (1) Token manipulation, (2) Gradient based attack, (3) Jailbreak prompting, (4) Human red-teaming, (5) Model red-teaming."},
    },
]

client.create_dataset(name=rag_dataset_name)
for ex in rag_examples:
    client.create_dataset_item(
        dataset_name=rag_dataset_name,
        id=ex["inputs"]["question"],
        input=ex["inputs"],
        expected_output=ex["outputs"],
    )

print(f"Created dataset '{rag_dataset_name}' with {len(rag_examples)} items")''')

md("""### Evaluators or Metrics
1. Correctness: Response vs reference answer
- Goal: Measure "how similar/correct is the RAG chain answer, relative to a ground-truth answer"
- Mode: Requires a ground truth (reference) answer supplied through a dataset
- Evaluator: Use LLM-as-judge to assess answer correctness.""")

code('''from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI

## Correctness output schema
class CorrectnessGrade(TypedDict):
    # Putting the explanation before the boolean forces the model to reason first.
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

correctness_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER.

Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer.
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

grader_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
    CorrectnessGrade, method="json_schema", strict=True
)

## evaluator - note: for the RAG task, `output` is the dict {"answer", "documents"}
def correctness(*, input, output, expected_output, metadata=None, **kwargs):
    """An evaluator for RAG answer accuracy"""
    answers = f"""\
QUESTION: {input['question']}
GROUND TRUTH ANSWER: {expected_output['answer']}
STUDENT ANSWER: {output['answer']}"""

    grade = grader_llm.invoke([
        {"role": "system", "content": correctness_instructions},
        {"role": "user", "content": answers},
    ])
    return Evaluation(name="correctness", value=grade["correct"], comment=grade["explanation"])''')

md("""### Relevance: Response vs input
The flow is similar to above, but we simply look at the inputs and outputs without needing the reference answer. Without a reference answer we can't grade accuracy, but can still grade relevance—as in, did the model address the user's question or not.""")

code('''# Grade output schema
class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "Provide the score on whether the answer addresses the question"]

relevance_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION and a STUDENT ANSWER.

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

relevance_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
    RelevanceGrade, method="json_schema", strict=True
)

def relevance(*, input, output, expected_output=None, metadata=None, **kwargs):
    """A simple evaluator for RAG answer helpfulness."""
    answer = f"QUESTION: {input['question']}\\nSTUDENT ANSWER: {output['answer']}"
    grade = relevance_llm.invoke([
        {"role": "system", "content": relevance_instructions},
        {"role": "user", "content": answer},
    ])
    return Evaluation(name="relevance", value=grade["relevant"], comment=grade["explanation"])''')

md("""### Groundedness: Response vs retrieved docs
Another useful way to evaluate responses without needing reference answers is to check if the response is justified by (or "grounded in") the retrieved documents.""")

code('''# Grade output schema
class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "Provide the score on if the answer hallucinates from the documents"]

grounded_instructions = """You are a teacher grading a quiz.

You will be given FACTS and a STUDENT ANSWER.

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS.
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

grounded_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
    GroundedGrade, method="json_schema", strict=True
)

def groundedness(*, input, output, expected_output=None, metadata=None, **kwargs):
    """A simple evaluator for RAG answer groundedness."""
    doc_string = "\\n\\n".join(doc.page_content for doc in output["documents"])
    answer = f"FACTS: {doc_string}\\nSTUDENT ANSWER: {output['answer']}"
    grade = grounded_llm.invoke([
        {"role": "system", "content": grounded_instructions},
        {"role": "user", "content": answer},
    ])
    return Evaluation(name="groundedness", value=grade["grounded"], comment=grade["explanation"])''')

md("### Retrieval Relevance: Retrieved docs vs input")

code('''# Grade output schema
class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the retrieved documents are relevant to the question, False otherwise"]

retrieval_relevance_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION and a set of FACTS provided by the student.

Here is the grade criteria to follow:
(1) Your goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.

Avoid simply stating the correct answer at the outset."""

retrieval_relevance_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
    RetrievalRelevanceGrade, method="json_schema", strict=True
)

def retrieval_relevance(*, input, output, expected_output=None, metadata=None, **kwargs):
    """An evaluator for document relevance"""
    doc_string = "\\n\\n".join(doc.page_content for doc in output["documents"])
    answer = f"FACTS: {doc_string}\\nQUESTION: {input['question']}"
    grade = retrieval_relevance_llm.invoke([
        {"role": "system", "content": retrieval_relevance_instructions},
        {"role": "user", "content": answer},
    ])
    return Evaluation(name="retrieval_relevance", value=grade["relevant"], comment=grade["explanation"])''')

md("### Run the evaluation")

code('''# The task wraps our traced rag_bot; its dict return is passed to every evaluator as `output`.
def rag_task(*, item, **kwargs) -> dict:
    return rag_bot(item.input["question"])

rag_dataset = client.get_dataset(rag_dataset_name)

result = client.run_experiment(
    name="rag-doc-relevance",
    data=rag_dataset.items,
    task=rag_task,
    evaluators=[correctness, groundedness, relevance, retrieval_relevance],
    metadata={"version": "LCEL context, gpt-4o-mini"},
)

print(result.format())

# Make sure everything is sent before the notebook exits.
client.flush()''')

nb = {
    "cells": cells,
    "metadata": {"language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1-rag_evaluation-todo.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("wrote", len(cells), "cells to", out)
