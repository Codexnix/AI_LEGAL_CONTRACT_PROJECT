"""Prompt templates for the AI Contract Analysis Pipeline.

This module is responsible solely for constructing the system and user
prompts sent to the LLM for clause extraction and contract
summarization. It performs no LLM calls, no PDF loading, no text
preprocessing, no clause extraction logic, and no response parsing —
it only produces prompt strings.
"""

from __future__ import annotations


class PromptBuilder:
    """Builds system and user prompts for contract analysis LLM calls.

    All methods are static: this class holds no instance state and
    exists purely to group related prompt-construction logic under one
    namespace.
    """

    @staticmethod
    def get_extraction_system_prompt() -> str:
        """Build the system prompt for the clause extraction task.

        Returns:
            A system prompt instructing the LLM to act as a precise,
            non-hallucinating legal contract analysis assistant.
        """
        return (
            "You are a meticulous legal contract analysis assistant. "
            "Your sole task is to extract specific clauses from the "
            "contract text provided by the user.\n\n"
            "You must follow these rules strictly:\n"
            "- Base your answer ONLY on the provided contract text. "
            "Do not use outside knowledge or assumptions.\n"
            "- Never hallucinate or invent clause text that does not "
            "appear in the contract.\n"
            "- If a requested clause is not present in the contract, "
            "return null for that clause. Do not guess or fabricate a "
            "substitute.\n"
            "- Accuracy takes priority over completeness. It is far "
            "better to return null than to return incorrect or "
            "invented information.\n"
            "- Extract the exact clause wording whenever possible, "
            "preserving the original wording from the contract "
            "verbatim.\n"
            "- Do not paraphrase, summarize, or rewrite the clause "
            "text in any way.\n"
            "- If a clause cannot be found in the contract, return "
            "null for it."
        )

    @staticmethod
    def build_extraction_prompt(contract_text: str) -> str:
        """Build the user prompt requesting extraction of three key clauses.

        Args:
            contract_text: The full contract text to extract clauses
                from.

        Returns:
            A user prompt instructing the LLM to return the
            termination, confidentiality, and liability clauses as
            strict JSON.

        Raises:
            ValueError: If ``contract_text`` is empty or contains only
                whitespace.
        """
        if not contract_text or not contract_text.strip():
            raise ValueError("contract_text must not be empty.")

        return (
            "Extract the following three clauses from the contract "
            "text below:\n"
            "1. Termination Clause\n"
            "2. Confidentiality Clause\n"
            "3. Liability Clause\n\n"
            "Respond with exactly ONE valid JSON object, matching "
            "exactly this schema:\n"
            "{\n"
            '  "termination_clause": "...",\n'
            '  "confidentiality_clause": "...",\n'
            '  "liability_clause": "..."\n'
            "}\n\n"
            "If a clause is not present in the contract, set its value "
            "to null.\n"
            "Strict output requirements:\n"
            "- Do not include markdown.\n"
            "- Do not include code fences.\n"
            "- Do not include explanations.\n"
            "- Do not include comments.\n"
            "- Do not include any text before the JSON object.\n"
            "- Do not include any text after the JSON object.\n"
            "- The response must be directly parseable by "
            "json.loads() with no modification.\n\n"
            "Contract text:\n"
            f"{contract_text}"
        )

    @staticmethod
    def get_summary_system_prompt() -> str:
        """Build the system prompt for the contract summarization task.

        Returns:
            A system prompt instructing the LLM to produce concise,
            objective, business-style summaries without legal advice.
        """
        return (
            "You are a business analyst assistant that writes concise, "
            "objective summaries of legal contracts for a non-legal "
            "business audience.\n\n"
            "You must follow these rules strictly:\n"
            "- Be objective and neutral in tone. Do not express opinions.\n"
            "- Base your summary ONLY on the provided contract text. "
            "Never hallucinate or invent details.\n"
            "- Do not provide legal advice, recommendations, or "
            "interpretations of legal validity.\n"
            "- If information is not explicitly stated in the "
            "contract, do not assume or fabricate it."
        )

    @staticmethod
    def build_summary_prompt(contract_text: str) -> str:
        """Build the user prompt requesting a concise contract summary.

        Args:
            contract_text: The full contract text to summarize.

        Returns:
            A user prompt instructing the LLM to produce a 100-150 word
            business-style summary of the contract.

        Raises:
            ValueError: If ``contract_text`` is empty or contains only
                whitespace.
        """
        if not contract_text or not contract_text.strip():
            raise ValueError("contract_text must not be empty.")

        return (
            "Write a concise summary of the contract below in 100 to "
            "150 words.\n\n"
            "The summary must:\n"
            "- State the purpose of the contract.\n"
            "- Mention the major parties involved, but only if they "
            "are explicitly named in the contract.\n"
            "- Mention the key obligations of the parties.\n"
            "- Mention the important risks or liabilities.\n"
            "- Not fabricate any information not present in the "
            "contract text.\n\n"
            "Return only the summary text, with no headings, labels, "
            "or additional commentary.\n\n"
            "Contract text:\n"
            f"{contract_text}"
        )