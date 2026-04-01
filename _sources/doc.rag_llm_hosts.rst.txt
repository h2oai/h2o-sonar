RAG and LLM Hosts
=================
H2O Sonar can evaluate standalone LLMs (Large Language Models) and LLMs used by RAG
(Retrieval-augmented generation) which are **hosted by**
the following products and services:

- :ref:`Amazon Bedrock`
- :ref:`Anthropic Claude Chat LLM Host`
- :ref:`Enterprise H2ogpte LLM Host`
- :ref:`H2o Gpt LLM Host`
- :ref:`H2o LLMops LLM Host`
- :ref:`Microsoft Azure Open Ai Chat LLM Host`
- :ref:`Ollama LLM Host`
- :ref:`Open Ai Assistants With File Search (Formerly Retrieval) Tool LLM Host`
- :ref:`Open Ai Chat Api Compatible LLM Host`
- :ref:`Open Ai Chat LLM Host`


Enterprise h2oGPTe LLM Host
---------------------------
The Enterprise h2oGPTe is RAG product that uses LLMs to generate responses.
H2O Sonar can be used to evaluate the performance of LLMs hosted by the Enterprise h2oGPTe.

Enterprise h2oGPTe LLM host connection configuration example:

.. code-block:: python

    h2o_gpte_connection = h2o_sonar_config.ConnectionConfig(
        connection_type=h2o_sonar_config.ConnectionConfigType.H2O_GPT_E.name,
        name="H2O GPT Enterprise",
        description="H2O GPT Enterprise LLM host example.",
        server_url="https://h2ogpte.h2o.ai/",
        token="YOUR_API_TOKEN_HERE",
        token_use_type=h2o_sonar_config.TokenUseType.API_KEY.name,
    )

Remarks:

- `connection_type` - `H2O_GPT_E`.
- `server_url` - URL of the Enterprise h2oGPTe LLM host.
- `token` - API key to access the Enterprise h2oGPTe LLM host - it can be generated in the Enterprise h2oGPTe UI (settings).
- `token_use_type` - API key type - `API_KEY`.

The following **model parameters** can be configured when building h2oGPTe :ref:`Test Lab`:

.. code-block:: json

    {
        "embedding_model": null,
        "prompt_template_id": null,
        "system_prompt": null,
        "pre_prompt_query": null,
        "prompt_query": null,
        "pre_prompt_summary": null,
        "prompt_summary": null,
        "llm": null,
        "llm_args": {
            "temperature": 0.0,
            "seed": 0,
            "top_k": 1,
            "top_p": 1.0,
            "repetition_penalty": 1.07,
            "max_new_tokens": 1024,
            "min_max_new_tokens": 512
        },
        "self_reflection_config": null,
        "rag_config": null,
        "timeout": null
    }


H2O GPT LLM Host
----------------
The H2O GPT is a product that is used to host LLMs. H2O Sonar can be used to evaluate the performance of LLMs hosted by the H2O GPT.

Enterprise H2O GPT LLM host connection configuration example:

.. code-block:: python

    h2o_gpt_connection = h2o_sonar_config.ConnectionConfig(
        connection_type=h2o_sonar_config.ConnectionConfigType.H2O_GPT.name,
        name="H2O GPT",
        description="H2O GPT LLM host example.",
        server_url="https://gpt.h2o.ai:5000/v1",
        token=os.getenv(KEY_H2OGPT_API_KEY),
        token_use_type=h2o_sonar_config.TokenUseType.API_KEY.name,
    )

Remarks:

- `connection_type` - `H2O_GPT`.
- `server_url` - URL of the H2O GPT LLM host.
- `token` - API key to access the H2O GPT LLM host.
- `token_use_type` - API key type - `API_KEY`.

The following **model parameters** can be configured when building h2oGPT :ref:`Test Lab`:

.. code-block:: json

    {
        "messages": null,
        "frequency_penalty": null,
        "function_call": null,
        "functions": null,
        "logit_bias": null,
        "logprobs": null,
        "max_tokens": null,
        "n": null,
        "presence_penalty": null,
        "response_format": null,
        "seed": null,
        "stop": null,
        "stream": null,
        "temperature": null,
        "tool_choice": null,
        "tools": null,
        "top_logprobs": null,
        "top_p": null,
        "user": null,
        "extra_headers": null,
        "extra_query": null,
        "extra_body": null,
        "timeout": null
    }


H2O LLMOps LLM Host
-------------------
The H2O LLMOps is a product that is used to host LLMs. H2O Sonar can be used to evaluate the performance of LLMs hosted by the H2O LLMOps.
LLMs can be deployed using the H2O LLMOps **deployer**.

.. code-block:: python

    h2o_llmops_connection = h2o_sonar_config.ConnectionConfig(
        connection_type=h2o_sonar_config.ConnectionConfigType.H2O_LLM_OPS.name,
        name="H2O LLMOps: h2o-danube-1.8b-chat",
        description="H2O LLMOps as host of h2o-danube-1.8b-chat LLM model.",
        server_url="https://model.h2o.ai/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/v1",
        token="YOUR_API_TOKEN_HERE",
        token_use_type=h2o_sonar_config.TokenUseType.API_KEY.name,
    )

Remarks:

- `connection_type` - `H2O_LLM_OPS`.
- `server_url` - URL of the OpenAI chat API endpoint created by the H2O LLMOps deployer.
- `token` - API key to access the LLM model.
- `token_use_type` - API key type - `API_KEY`.

The following **model parameters** can be configured when building :ref:`Test Lab`:

.. code-block:: json

    {
        "messages": null,
        "frequency_penalty": null,
        "function_call": null,
        "functions": null,
        "logit_bias": null,
        "logprobs": null,
        "max_tokens": null,
        "n": null,
        "presence_penalty": null,
        "response_format": null,
        "seed": null,
        "stop": null,
        "stream": null,
        "temperature": null,
        "tool_choice": null,
        "tools": null,
        "top_logprobs": null,
        "top_p": null,
        "user": null,
        "extra_headers": null,
        "extra_query": null,
        "extra_body": null,
        "timeout": null
    }


Open AI Assistants with File Search (formerly Retrieval) Tool LLM Host
----------------------------------------------------------------------
Open AI Assistants with the `File Search <https://platform.openai.com/docs/assistants/tools/file-search>`_ tool
(or deprecated ``Retrieval`` tool) is RAG system from OpenAI which hosts LLMs to generate the answers.
H2O Sonar can be used to evaluate the performance of LLMs hosted by Open AI Assistants with the tools.

Open AI Assistants with ``Retrieval`` tool is available when ``openai`` client library version ``1.20``
and below is installed. Open AI Assistants with the ``File Search`` tool is available when
newer ``openai`` client library is installed. H2O Sonar automatically detects the tool availability
and uses the appropriate LLM/RAG client and tool when connecting to the OpenAI Assistants.

However, there are important **limitations** when using the OpenAI Assistants:

* The OpenAI Assistants **version 2** with the ``File Search`` tool do **not** provide retrieved contexts
  when H2O Sonar builds the :ref:`test lab<Test Lab>`. Therefore the ``retrieved_contexts`` field in the test lab
  will be empty and :ref:`evaluators<Evaluators>` which require the retrieved contexts should **not**
  be used as they will not work as expected - their results will be based on the generated responses only
  and might be incorrect. H2O Sonar will generate problems for the test lab with the empty retrieved
  contexts.
* The OpenAI Assistants **version 1** with the ``Retrieval`` tool is **deprecated** and will be removed in the future.
  OpenaAI's endpoint provided the the retrieved context in the past. However, as part of the deprecation
  process the retrieved context is no longer provided by the OpenAI's endpoint as well which brings
  evaluators accuracy issues described above.

.. code-block:: python

    openai_rag_connection = h2o_sonar_config.ConnectionConfig(
        connection_type=h2o_sonar_config.ConnectionConfigType.OPENAI_RAG.name,
        name="OpenAI RAG",
        description="OpenAI AI Assistant with the tool enabled.",
        token=os.getenv(KEY_OPENAI_API_KEY),
        token_use_type=h2o_sonar_config.TokenUseType.API_KEY.name,
    )

Remarks:

- `connection_type` - `OPENAI_RAG`.
- `server_url` is resolved internally by the client.
- `token` - API key to access the LLM model.
- `token_use_type` - API key type - `API_KEY`.

The following **model parameters** can be configured when building OpenAI :ref:`Test Lab`:

.. code-block:: json

    {
        "assistant_kwargs": {
            "name": null,
            "description": null,
            "instructions": null,
            "tools": null,
            "metadata": null,
            "extra_headers": null,
            "extra_query": null,
            "extra_body": null,
            "timeout": null
        },
        "thread_kwargs": {
            "messages": null,
            "metadata": null,
            "extra_headers": null,
            "extra_query": null,
            "extra_body": null,
            "timeout": null
        },
        "run_kwargs": {
            "additional_instructions": null,
            "additional_messages": null,
            "instructions": null,
            "max_completion_tokens": null,
            "max_prompt_tokens": null,
            "metadata": null,
            "response_format": null,
            "stream": null,
            "temperature": null,
            "tool_choice": null,
            "tools": null,
            "truncation_strategy": null,
            "extra_headers": null,
            "extra_query": null,
            "extra_body": null,
            "timeout": null
        }
    }


Open AI Chat LLM Host
---------------------
Open AI Chat is a product that hosts LLMs. H2O Sonar can be used to evaluate the performance of LLMs hosted by the Open AI Chat.

.. code-block:: python

    openai_connection = h2o_sonar_config.ConnectionConfig(
        connection_type=h2o_sonar_config.ConnectionConfigType.OPENAI_CHAT.name,
        name="OpenAI Chat",
        description="OpenAI chat API.",
        token=os.getenv(KEY_OPENAI_API_KEY),
        token_use_type=h2o_sonar_config.TokenUseType.API_KEY.name,
    )

Remarks:

- `connection_type` - `OPENAI_CHAT`.
- `server_url` is resolved internally by the client.
- `token` - API key to access the LLM model.
- `token_use_type` - API key type - `API_KEY`.

The following **model parameters** can be configured when building :ref:`Test Lab`:

.. code-block:: json

    {
        "messages": null,
        "frequency_penalty": null,
        "function_call": null,
        "functions": null,
        "logit_bias": null,
        "logprobs": null,
        "max_tokens": null,
        "n": null,
        "presence_penalty": null,
        "response_format": null,
        "seed": null,
        "stop": null,
        "stream": null,
        "temperature": null,
        "tool_choice": null,
        "tools": null,
        "top_logprobs": null,
        "top_p": null,
        "user": null,
        "extra_headers": null,
        "extra_query": null,
        "extra_body": null,
        "timeout": null
    }


Microsoft Azure Open AI Chat LLM Host
-------------------------------------
Microsoft Azure hosted Open AI Chat is a product that hosts LLMs.
H2O Sonar can be used to evaluate the performance of LLMs hosted by the Open AI Chat.

.. code-block:: python

    openai_connection = h2o_sonar_config.ConnectionConfig(
        connection_type=h2o_sonar_config.ConnectionConfigType.AZURE_OPENAI_CHAT.name,
        name="OpenAI Chat at MS Azure",
        description="OpenAI chat API hosted by Microsoft Azure.",
        server_url="https://my-llm-environment.openai.azure.com/",
        server_id="my-llm-testing",
        token=os.getenv(KEY_AZURE_OPENAI_API_KEY),
        token_use_type=h2o_sonar_config.TokenUseType.API_KEY.name,
    )

Remarks:

- `connection_type` - `OPENAI_CHAT`.
- `server_url` is URL of the Open AI Chat environment hosted by Microsoft Azure.
- `server_id` - ID of the Open AI Chat environment hosted by Microsoft Azure used as the **LLM model name**.
- `token` - API key to access the environment.
- `token_use_type` - API key type - `API_KEY`.

The following **model parameters** can be configured when building Microsoft Azure :ref:`Test Lab`:

.. code-block:: json

    {
        "messages": null,
        "frequency_penalty": null,
        "function_call": null,
        "functions": null,
        "logit_bias": null,
        "logprobs": null,
        "max_tokens": null,
        "n": null,
        "presence_penalty": null,
        "response_format": null,
        "seed": null,
        "stop": null,
        "stream": null,
        "temperature": null,
        "tool_choice": null,
        "tools": null,
        "top_logprobs": null,
        "top_p": null,
        "user": null,
        "extra_headers": null,
        "extra_query": null,
        "extra_body": null,
        "timeout": null
    }


Anthropic Claude Chat LLM Host
-------------------------------
Anthropic Claude Chat is a product that hosts Claude LLMs. H2O Sonar can be used to evaluate the performance of Claude models hosted by Anthropic.

.. code-block:: python

    anthropic_connection = h2o_sonar_config.ConnectionConfig(
        connection_type=h2o_sonar_config.ConnectionConfigType.ANTHROPIC_CHAT.name,
        name="Anthropic Claude Chat",
        description="Anthropic Claude AI chat API.",
        token=os.getenv(KEY_ANTHROPIC_API_KEY),
        token_use_type=h2o_sonar_config.TokenUseType.API_KEY.name,
    )

Remarks:

- `connection_type` - `ANTHROPIC_CHAT`.
- `server_url` is resolved internally by the client.
- `token` - API key to access the Anthropic Claude API.
- `token_use_type` - API key type - `API_KEY`.

The following **model parameters** can be configured when building Anthropic :ref:`Test Lab`:

.. code-block:: json

    {
        "max_tokens": null,
        "metadata": null,
        "stop_sequences": null,
        "system": null,
        "temperature": null,
        "tool_choice": null,
        "tools": null,
        "top_k": null,
        "top_p": null,
        "extra_headers": null,
        "extra_query": null,
        "extra_body": null,
        "timeout": null
    }


Open AI Chat API Compatible LLM Host
------------------------------------
H2O Sonar can be used to evaluate the performance of LLMs hosted by any OpenAI API compatible LLM host.

.. code-block:: python

    openai_connection = h2o_sonar_config.ConnectionConfig(
        connection_type=h2o_sonar_config.ConnectionConfigType.OPENAI_CHAT.name,
        name="OpenAI Chat",
        description="OpenAI chat API.",
        server_url="https://model.h2o.ai/0ac9b9c8-91f3-485c-bc1c-17163c5d75b5/v1",
        token=os.getenv(KEY_OPENAI_API_KEY),
        token_use_type=h2o_sonar_config.TokenUseType.API_KEY.name,
    )

Remarks:

- `connection_type` - `OPENAI_CHAT`.
- `server_url` - URL of the Open AI Chat compatible endpoint.
- `token` - API key to access the LLM model.
- `token_use_type` - API key type - `API_KEY`.

The following **model parameters** can be configured when building :ref:`Test Lab`:

.. code-block:: json

    {
        "messages": null,
        "frequency_penalty": null,
        "function_call": null,
        "functions": null,
        "logit_bias": null,
        "logprobs": null,
        "max_tokens": null,
        "n": null,
        "presence_penalty": null,
        "response_format": null,
        "seed": null,
        "stop": null,
        "stream": null,
        "temperature": null,
        "tool_choice": null,
        "tools": null,
        "top_logprobs": null,
        "top_p": null,
        "user": null,
        "extra_headers": null,
        "extra_query": null,
        "extra_body": null,
        "timeout": null
    }


Amazon Bedrock
--------------

The current implementation of the Amazon Bedrock client supports RAG either with a predefined collection ID
that corresponds to ``knowledgeBaseId`` or it can create a knowledge base using some fixed configuration which is not configurable as of now.
Only Anthropic Claude models are `supported for usage in the RAG <https://docs.aws.amazon.com/bedrock/latest/userguide/models-features.html>`_.

.. code-block:: python

    bedrock_connection = h2o_sonar_config.ConnectionConfig(
        connection_type=h2o_sonar_config.ConnectionConfigType.AMAZON_BEDROCK.name,
        name="Amazon Bedrock",
        description="Amazon Bedrock RAG host connection.",
        username=os.getenv("AWS_ACCESS_KEY"),
        password=os.getenv("AWS_SECRET_ACCESS_KEY"),
        token=os.getenv("AWS_SESSION_TOKEN"),
    )

Remarks:

- `connection_type` - `AMAZON_BEDROCK`.
- `username` - AWS access key.
- `password` - AWS secret access key.
- `token` - AWS session token.

The following **model parameters** can be configured when building :ref:`Test Lab`:

.. code-block:: text

    {
         "guardrailConfiguration": {
            "guardrailId": "string",
            "guardrailVersion": "string"
         },
         "inferenceConfig": {
            "textInferenceConfig": {
               "maxTokens": number,
               "stopSequences": [ "string" ],
               "temperature": number,
               "topP": number
            }
         },
         "promptTemplate": {
            "textPromptTemplate": "string"
         },
         "orchestrationConfiguration": {
                "queryTransformationConfiguration": {
                    "type": "QUERY_DECOMPOSITION"
                }
         },
         "retrievalConfiguration": {
            "vectorSearchConfiguration": {
               "filter": { ... },
               "numberOfResults": number,
               "overrideSearchType": "string"
            }
         }
     }

These parameters are described in the `boto3 documentation <https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent-runtime/client/retrieve_and_generate.html>`_.
``guardrailConfiguration``, ``inferenceConfig``, ``promptTemplate`` are passed to the ``knowledgeBaseConfiguration``.


ollama LLM Host
---------------
H2O Sonar can be used to evaluate the performance of LLMs hosted by `ollama <https://ollama.com/>`_ LLM hosts.

.. code-block:: python

    ollama_connection = h2o_sonar_config.ConnectionConfig(
        connection_type=h2o_sonar_config.ConnectionConfigType.OLLAMA.name,
        name="ollama",
        description="ollama host LLM models.",
        server_url="http://localhost:11434",
    )

Remarks:

- `connection_type` - `OLLAMA`.
- `server_url` - URL of the `ollama` endpoint.

The following **model parameters** can be configured when building ollama :ref:`Test Lab`:

.. code-block:: json

    {
        "images": null,
        "options": {
            "num_ctx": 4096,
            "repeat_last_n": 64,
            "repeat_penalty": 1.1,
            "temperature": 0.7,
            "seed": 42,
            "stop": null,
            "tfs_z": 1.0,
            "num_predict": 128,
            "top_k": 40,
            "top_p": 0.9
        },
        "system": null,
        "context": null,
        "raw": false
    }


See also:

- :ref:`Connection Configuration`


