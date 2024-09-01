# import logging
# import sqlite3
import logging
import time
from http import HTTPStatus

import dashscope
import fitz
from elasticsearch import Elasticsearch
from elasticsearch import exceptions as es_exceptions
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
dashscope.api_key = "sk-b6b41dbe5d0b4b8591c2486271c9e18c"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Elasticsearch setup
ELASTICSEARCH_HOST = (
    "http://es-sg-ju33w74pb0002nt56.public.elasticsearch.aliyuncs.com:9200"
)
ELASTICSEARCH_USERNAME = "elastic"
ELASTICSEARCH_PASSWORD = "100%Winrate"

es = Elasticsearch(
    [ELASTICSEARCH_HOST], basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD)
)

# Define the embedding dimension to match the model output
EMBEDDING_DIM = 1024

# In-memory storage for chat history
chat_histories = {}

def create_question_answering_index():
    index_body = {
        "settings": {
            "index": {
                "number_of_shards": 2,
                "number_of_replicas": 1,
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM,  # Set to match the embedding model's dimension
                    "index": True,
                    "similarity": "l2_norm",
                },
                "content": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                },
                "filename": {
                    "type": "text",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                },
            }
        },
    }

    if not es.indices.exists(index="pdf-files"):
        es.indices.create(index="pdf-files", body=index_body)
        logger.info("Index for question answering created with the correct dimensions.")
    else:
        logger.info("Index already exists, skipping creation.")


create_question_answering_index()


def get_embedding(text):
    if len(text) < 1 or len(text) > 6000:
        logger.error(f"Input length is out of the allowed range [1, 6000]: {len(text)}")
        return None

    resp = dashscope.TextEmbedding.call(
        model=dashscope.TextEmbedding.Models.text_embedding_v3,
        input=text,
        dimension=EMBEDDING_DIM,  # Ensure this matches the dimension defined in your Elasticsearch index mapping
    )
    if resp.status_code == HTTPStatus.OK:
        return resp.output["embeddings"][0]["embedding"]
    else:
        logger.error(f"Embedding API call failed: {resp}")
        return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    user_message = (
        request.form.get("message") if request.form else request.data.decode("utf-8")
    )

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    session_id = request.cookies.get("session_id", "default")

    if session_id not in chat_histories:
        chat_histories[session_id] = []

    try:
        logger.info(f"Searching for relevant content for the query: {user_message}")

        embedding = get_embedding(user_message)

        if embedding is None:
            return jsonify({"error": "Failed to generate embedding."}), 500

        search_response = es.search(
            index="pdf-files",
            body={
                "_source": ["filename", "content"],
                "size": 5,
                "query": {
                    "knn": {
                        "field": "embedding",
                        "query_vector": embedding,
                        "num_candidates": 10,
                    }
                },
            },
        )

        hits = search_response.get("hits", {}).get("hits", [])

        references = {}
        retrieved_content = []

        for hit in hits:
            filename = hit["_source"]["filename"]
            content_snippet = hit["_source"]["content"][
                :500
            ]  # Limit to first 500 chars

            if filename not in references:
                references[filename] = len(references) + 1

            retrieved_content.append(
                f"Source [{references[filename]}]: {content_snippet}"
            )

        if not retrieved_content:
            retrieved_content = ["No relevant content found."]

        reference_list = [f"[{i}] {filename}" for filename, i in references.items()]

        combined_input = f"""Retrieved content:
{' '.join(retrieved_content)}

User question: {user_message}

Instructions:
1. Provide a detailed and comprehensive answer to the user's question.
2. Use ONLY the information from the provided sources.
3. Use in-text citations [1], [2], etc., referencing ONLY the following sources:
{'; '.join(f'[{references[filename]}] {filename}' for filename in references)}
4. Do NOT use any citations that are not in the above list.
5. If you need to mention information not from these sources, clearly state it's your own analysis or general knowledge.
6. Include a 'References:' section at the end listing only the sources you actually cited in your response.
7. Aim for a thorough response that fully addresses the user's query while strictly adhering to these citation rules."""

        chat_histories[session_id].append({"role": "user", "content": user_message})
        chat_histories[session_id].append({"role": "system", "content": combined_input})

        response = dashscope.Generation.call(
            "qwen-max",
            messages=chat_histories[session_id],
            result_format="message",
            stream=False,
            # max_tokens=2000,
        )

        if response.status_code == HTTPStatus.OK:
            bot_message = response.output.choices[0]["message"]["content"]

            chat_histories[session_id].append(
                {"role": "assistant", "content": bot_message}
            )

            # Ensure the bot's message ends with a newline before adding references
            if not bot_message.endswith("\n"):
                bot_message += "\n"

            # Add references if they're not already included in the bot's message
            if "References:" not in bot_message:
                bot_message += "\nReferences:\n" + "\n".join(reference_list)

            return jsonify({"botMessage": bot_message})
        else:
            logger.error(
                f"Request id: {response.request_id}, Status code: {response.status_code}, error code: {response.code}, error message: {response.message}"
            )
            return (
                jsonify(
                    {
                        "error": "An error occurred with the AI service. Please try again later."
                    }
                ),
                500,
            )

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred."}), 500


def split_into_chunks(text, max_length=512):
    sentences = text.split(".")
    chunks = []
    chunk = ""

    for sentence in sentences:
        if len(chunk) + len(sentence) + 1 > max_length:
            chunks.append(chunk.strip())
            chunk = sentence + "."
        else:
            chunk += sentence + "."

    if chunk:
        chunks.append(chunk.strip())

    return chunks


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if not es:
        return jsonify({"error": "Elasticsearch is not available"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith(".pdf"):
        try:
            text_extraction_start = time.time()

            # Extract text from the PDF
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()

            text_extraction_time = time.time() - text_extraction_start
            logger.info(f"Text extraction time: {text_extraction_time:.2f} seconds")

            # Split the text into smaller chunks
            chunks = split_into_chunks(text)

            # Process and index each chunk separately
            for chunk in chunks:
                # Embed the chunk using Alibaba Cloud Model Studio
                embedding = get_embedding(chunk)

                if embedding is None:
                    return jsonify({"error": "Failed to generate embedding."}), 500

                embedding_time = time.time() - text_extraction_start
                logger.info(f"Chunk embedding time: {embedding_time:.2f} seconds")

                # Index the chunk with its embedding
                try:
                    response = es.index(
                        index="pdf-files",
                        document={
                            "content": chunk,
                            "filename": file.filename,
                            "embedding": embedding,  # Use the embedding from Alibaba Cloud Model Studio
                        },
                    )
                    logger.info(
                        f"Indexed chunk: {chunk[:30]}..."
                    )  # Log a snippet of the chunk

                except es_exceptions.RequestError as e:
                    logger.error(f"Elasticsearch indexing error: {str(e)}")
                    return (
                        jsonify(
                            {"error": "Failed to index the chunk in Elasticsearch."}
                        ),
                        500,
                    )

            total_process_time = text_extraction_time + embedding_time
            logger.info(f"Total process time: {total_process_time:.2f} seconds")

            return jsonify(
                {
                    "message": "File uploaded and indexed successfully.",
                }
            )

        except fitz.FileDataError as e:
            logger.error(f"PDF processing error: {str(e)}")
            return (
                jsonify(
                    {
                        "error": "Failed to process the PDF file. The file might be corrupt."
                    }
                ),
                400,
            )
        except Exception as e:
            logger.error(f"Unexpected error while processing the file: {str(e)}")
            return (
                jsonify(
                    {"error": "An unexpected error occurred while processing the file."}
                ),
                500,
            )

    return jsonify({"error": "Invalid file type. Only PDF is allowed."}), 400


if __name__ == "__main__":
    app.run(port=5500, debug=True)
