from langchain.llms import VLLMOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.evaluation import load_evaluator
from langchain.embeddings import HuggingFaceEmbeddings

import requests
import json
import time

max_response_time = 3

INFERENCE_SERVER_URL = "https://parasol-instruct-claimsbot-ai.apps.cluster-jxp8q.sandbox1291.opentlc.com"
MAX_NEW_TOKENS = 512
TOP_P = 0.95
TEMPERATURE = 0.01
PRESENCE_PENALTY = 1.03

def infer_with_template(input_text, template):
    llm = VLLMOpenAI(
        openai_api_key="EMPTY",
        openai_api_base= f"{INFERENCE_SERVER_URL}/v1",
        model_name="parasol-instruct",
        max_tokens=MAX_NEW_TOKENS,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        presence_penalty=PRESENCE_PENALTY,
        streaming=False,
        verbose=False,
    )

    PROMPT = PromptTemplate.from_template(template)

    llm_chain = LLMChain(llm=llm, prompt=PROMPT)

    return llm_chain.run(input_text)

def similarity_metric(predicted_text, reference_text):
    embedding_model = HuggingFaceEmbeddings()
    evaluator = load_evaluator("embedding_distance", embeddings=embedding_model)
    distance_score = evaluator.evaluate_strings(prediction=predicted_text, reference=reference_text)
    return 1-distance_score["score"]



def send_request(endpoint):
    response = requests.get(endpoint)
    return response

def test_responsetime():
    TEMPLATE = """<s>[INST] <<SYS>>
Answer below truthfully and in less than 10 words:
<</SYS>>
{silly_question}
[/INST]"""
    
    start = time.perf_counter()
    response = infer_with_template("Who saw a saw saw a salsa?", TEMPLATE)
    response_time = time.perf_counter() - start

    if response_time>max_response_time:
        raise Exception(f"Response took {response_time} which is greater than {max_response_time}")

    print(f"Response time was OK at {response_time} seconds")

    with open("responsetime_result.json", "w") as f:
        json.dump({
            "response_time": response_time
        }, f)

if __name__ == '__main__':
    test_responsetime()