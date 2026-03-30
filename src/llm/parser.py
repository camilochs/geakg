"""Response parser for LLM outputs.

Parses LLM responses to extract structured operation proposals.
Uses multiple parsing strategies for robustness.
"""

import re
from typing import Any

from pydantic import BaseModel, Field

from src.constraints.validator import ProposedOperation


class ParseResult(BaseModel):
    """Result of parsing LLM response."""

    success: bool
    operation: ProposedOperation | None = None
    raw_response: str = ""
    parse_method: str = ""
    error: str | None = None


class ResponseParser:
    """Parser for LLM responses.

    Uses multiple parsing strategies:
    1. Structured format (Operation: X, Reasoning: Y)
    2. JSON extraction
    3. Fuzzy matching against known operators
    """

    def __init__(self, valid_operators: list[str]) -> None:
        """Initialize parser.

        Args:
            valid_operators: List of valid operator IDs for fuzzy matching
        """
        self.valid_operators = set(valid_operators)
        self._operator_pattern = re.compile(
            r"Operation:\s*([a-z_0-9]+)", re.IGNORECASE
        )
        self._reasoning_pattern = re.compile(
            r"Reasoning:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL
        )

    def parse(self, response: str) -> ParseResult:
        """Parse LLM response to extract operation.

        Args:
            response: Raw LLM response text

        Returns:
            ParseResult with extracted operation or error
        """
        response = response.strip()

        # Try structured format first
        result = self._parse_structured(response)
        if result.success:
            return result

        # Try JSON format
        result = self._parse_json(response)
        if result.success:
            return result

        # Try fuzzy matching
        result = self._parse_fuzzy(response)
        if result.success:
            return result

        return ParseResult(
            success=False,
            raw_response=response,
            error="Could not extract operation from response",
        )

    def _parse_structured(self, response: str) -> ParseResult:
        """Parse structured format: Operation: X, Reasoning: Y.

        Args:
            response: Raw response

        Returns:
            ParseResult
        """
        op_match = self._operator_pattern.search(response)
        if not op_match:
            return ParseResult(
                success=False,
                raw_response=response,
                parse_method="structured",
                error="No 'Operation:' pattern found",
            )

        operation_id = op_match.group(1).lower().strip()

        # Extract reasoning if present
        reasoning = None
        reason_match = self._reasoning_pattern.search(response)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        # Validate operation exists
        if operation_id not in self.valid_operators:
            # Try to find closest match
            closest = self._find_closest_operator(operation_id)
            if closest:
                operation_id = closest
            else:
                return ParseResult(
                    success=False,
                    raw_response=response,
                    parse_method="structured",
                    error=f"Unknown operator: {operation_id}",
                )

        return ParseResult(
            success=True,
            operation=ProposedOperation(
                operation_id=operation_id,
                reasoning=reasoning,
            ),
            raw_response=response,
            parse_method="structured",
        )

    def _parse_json(self, response: str) -> ParseResult:
        """Try to extract JSON from response.

        Args:
            response: Raw response

        Returns:
            ParseResult
        """
        import json

        # Try to find JSON block
        json_pattern = re.compile(r"\{[^{}]+\}", re.DOTALL)
        matches = json_pattern.findall(response)

        for match in matches:
            try:
                data = json.loads(match)
                if "operation" in data or "operation_id" in data:
                    op_id = data.get("operation_id") or data.get("operation")
                    op_id = op_id.lower().strip()

                    if op_id in self.valid_operators:
                        return ParseResult(
                            success=True,
                            operation=ProposedOperation(
                                operation_id=op_id,
                                reasoning=data.get("reasoning"),
                            ),
                            raw_response=response,
                            parse_method="json",
                        )
            except json.JSONDecodeError:
                continue

        return ParseResult(
            success=False,
            raw_response=response,
            parse_method="json",
            error="No valid JSON with operation found",
        )

    def _parse_fuzzy(self, response: str) -> ParseResult:
        """Fuzzy match operators mentioned in response.

        Args:
            response: Raw response

        Returns:
            ParseResult
        """
        response_lower = response.lower()

        # Look for any operator mentioned in the response
        found_operators = []
        for op in self.valid_operators:
            if op in response_lower:
                # Find position to prioritize earlier mentions
                pos = response_lower.index(op)
                found_operators.append((op, pos))

        if found_operators:
            # Take the first mentioned operator
            found_operators.sort(key=lambda x: x[1])
            best_op = found_operators[0][0]

            return ParseResult(
                success=True,
                operation=ProposedOperation(
                    operation_id=best_op,
                    reasoning="(extracted via fuzzy matching)",
                ),
                raw_response=response,
                parse_method="fuzzy",
            )

        return ParseResult(
            success=False,
            raw_response=response,
            parse_method="fuzzy",
            error="No known operators found in response",
        )

    def _find_closest_operator(self, candidate: str) -> str | None:
        """Find closest matching operator.

        Args:
            candidate: Candidate operator string

        Returns:
            Closest valid operator or None
        """
        # Check for common variations
        variations = [
            candidate,
            candidate.replace("-", "_"),
            candidate.replace(" ", "_"),
            candidate + "_opt",
        ]

        for var in variations:
            if var in self.valid_operators:
                return var

        # Check substring matches
        for op in self.valid_operators:
            if candidate in op or op in candidate:
                return op

        return None

    def update_valid_operators(self, operators: list[str]) -> None:
        """Update the list of valid operators.

        Args:
            operators: New list of valid operator IDs
        """
        self.valid_operators = set(operators)
