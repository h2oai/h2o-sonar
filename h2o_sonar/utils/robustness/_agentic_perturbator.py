# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.
import json
import traceback

from h2o_sonar import errors
from h2o_sonar import loggers
from h2o_sonar.lib.api import agents
from h2o_sonar.lib.api import commons


class AbcAgenticPerturbator:
    """Abstract class for agentic and LLM perturbators which can use h2oGPTe agent
    API to perturb the text:

    - provides h2oGPTe agent API connection w/ health check
      - health check is NOT done on listing / registration / ... to avoid
        hangs of server(s) where is H2O Soar used
    - default prompt template + agent response processing/parsing/extraction
    - perturbs 1 or more prompts
    - child classes need to specify:
       - agent prompt

    Given the perturbators API design (all other) ONE prompt is perturbed at a time,
    in other words there is not BULK perturbation of prompts and the perturbation
    is slow and expensive.

    As a side effect this class can server both as LLM and AGENTIC perturbator.

    """

    def __init__(
        self,
        instructions: str,
        example_text: str,
        example_perturbed_text: str,
        llm_only: bool = False,
        logger: loggers.SonarLogger | None = None,
        log_name: str | None = "AgenticPerturbator",
        **extra_params,
    ):
        """Initialize the agentic perturbator.

        Parameters
        ----------
        instructions : str
            Instructions for the perturbation.
        example_input_text : str
            Example of the input text.
        example_perturbed_text : str
            Example of the perturbed text.
        llm_only : bool
            Whether to use only LLM, not agent for the perturbations.
        logger : loggers.SonarLogger | None
            Logger to use.
        log_name : str | None
            Name of the logger.

        """
        self._agent_host = None  # None not initialized, object initialized, str error

        self.instructions = instructions
        self.example_text = example_text
        self.example_perturbed_text = example_perturbed_text

        self.llm_only = llm_only

        self.logger = logger or loggers.SonarPrintLogger()
        self.log_name = log_name

        self.extra_params = extra_params

    def check_compatibility(self):
        """Check whether the agent host is configured - the check aims to be fast and
        non-blocking/non-hanging. Therefore, health check is not done.

        """
        try:
            return (
                True
                if agents.H2oGpteAgentHost(
                    llm_only=self.llm_only, logger=self.logger, log_name=self.log_name
                ).agent_connection
                else False
            )
        except Exception as e:
            self.logger.error(
                f"Error while checking {self.log_name} compatibility: {e}"
            )
            return False

    KEY_INPUT_TEXT = "input_text"
    KEY_PERTURBED_TEXT = "perturbed_text"

    def create_prompt(
        self,
        text: str,
        instructions: str = "",
        example_text: str = "",
        example_perturbed_text: str = "",
    ) -> str:
        """Bind prompt template with parameters and return the prompt."""
        instructions = instructions or self.instructions
        example_text = example_text or self.example_text
        example_perturbed_text = example_perturbed_text or self.example_perturbed_text

        prompt = f"""
Your task is to perturb INPUT TEXT.

Instructions for how to perturb the INPUT TEXT:

[BEGIN PERTURBATION INSTRUCTIONS]
{instructions}
[END PERTURBATION INSTRUCTIONS]

Instructions how to return the perturbed result:

- provide the perturbed text as JSon with the following structure:

    {{
        "{AbcAgenticPerturbator.KEY_INPUT_TEXT}": string,
        "{AbcAgenticPerturbator.KEY_PERTURBED_TEXT}": string,
    }}

- input_text: is the INPUT TEXT which was perturbed
- perturbed_text: is the perturbed text you created

Example of the perturbed text:

    {{
        "input_text": "{example_text}",
        "perturbed_text": "{example_perturbed_text}",
    }}

INPUT TEXT data:

[BEGIN INPUT_TEXT]
{text}
[END INPUT_TEXT]
"""

        return prompt

    def _extract_answer(self, agent_answer: str) -> str | None:
        """Strive to get the perturbed text from the agent's answer.

        Returns
        -------
        str | None
            Perturbed text.

        """
        self.logger.info(
            f"{self.log_name}: parsing agent's answer: >>>{agent_answer}<<< to "
            f"get the perturbed text..."
        )

        if not agent_answer or not isinstance(agent_answer, str):
            return None

        # check whether the answer contains the expected JSon keys
        if self.KEY_PERTURBED_TEXT not in agent_answer:
            return None

        # parse JSon created by the agent
        perturbed_text = None
        try:
            json_start = agent_answer.find("{")
            json_end = agent_answer.rfind("}")
            if json_start == -1 or json_end == -1:
                return None

            a_dict = json.loads(agent_answer[json_start : json_end + 1])

            perturbed_text = a_dict.get(self.KEY_PERTURBED_TEXT)
        except Exception as ex:
            self.logger.warning(
                f"{self.log_name}: unable to parse agent's JSon answer from "
                f">>>{agent_answer}<<<: "
                f"{ex}\n{traceback.format_exc()}"
            )

        return perturbed_text

    def agent_perturb(
        self,
        text: str,
        intensity: commons.PerturbationIntensity,
        raised_errors: list | None = None,
        **kwargs,
    ) -> str | None:
        """Perturb the text using the agent API/LLM.

        Parameters
        ----------
        text : str
            Text to perturb.
        intensity : commons.PerturbationIntensity
            Intensity of the perturbation.
        raised_errors : list | None
            List to store raised errors.

        Returns
        -------
        str | None
            Perturbed text.

        """
        del intensity

        if not text:
            return text

        err_msg = None
        try:
            if self._agent_host is None:
                if self.check_compatibility():
                    self._agent_host = agents.H2oGpteAgentHost(
                        llm_only=self.llm_only,
                        logger=self.logger,
                        log_name=self.log_name,
                    )
                else:
                    self._agent_host = "agent host not configured / initialized"
                    err_msg = (
                        f"Perturbator '{self.log_name}' failed to produce perturbed "
                        f"text as the agent host (which is needed) is not configured "
                        f"or initialized."
                    )
            elif isinstance(self._agent_host, str):
                err_msg = (
                    f"Perturbator '{self.log_name}' failed to produce perturbed "
                    f"text as the agent host - which is needed - is not available: "
                    f"{self._agent_host}"
                )

            #
            # agentic perturbation
            #

            if err_msg is None:
                agentic_prompts = [self.create_prompt(text=text)]

                # hope for the best
                agentic_answers = self._agent_host.ask_agent(agentic_prompts)
                self.logger.info(f"Agentic perturbation answers: {agentic_answers}")
                if (
                    not agentic_answers
                    or len(agentic_answers) < 1
                    or not agentic_answers[0]
                ):
                    err_msg = (
                        f"Perturbator '{self.log_name}' failed to produce perturbed "
                        f"text as the agent did not return the any answer."
                    )
                else:
                    perturbed_text = self._extract_answer(agentic_answers[0].answer)
                    if perturbed_text:
                        return str(perturbed_text)
                    else:
                        err_msg = (
                            f"Perturbator '{self.log_name}' failed to produce "
                            f"perturbed text as it was not able to parse the agent's "
                            f"answer and extract the result."
                        )
        except Exception as e:
            err_msg = (
                f"Perturbator '{self.log_name}' failed to produce perturbed text: {e}"
            )

        if raised_errors is not None:
            raised_errors.append(err_msg)
            return None
        raise errors.MliError(err_msg)
