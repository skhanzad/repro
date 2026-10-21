"""Decide whether a GitHub issue is a bug report, as cheaply as possible.

Three tiers, each only reached when the previous one can't decide:

1. Rules (free): issue labels and title tags like "[Bug]" or "[Feature Request]".
   Conflicting or missing signals fall through.
2. Cheap model: title + truncated body -> category, confidence, and whether the
   issue is ambiguous (could reasonably be filed under another category).
3. Escalation model: re-asks a stronger model when the cheap one is unsure: it flags
   the issue as ambiguous, or its confidence is below `escalate_below`. Disabled when
   no escalation model is set. Small models report high confidence even when wrong,
   so the ambiguity flag is usually the more useful trigger.

Results are cached by issue key and a hash of the issue's content, so an issue is
only classified again when its title, body or labels change.
"""

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from repro.config import Provider, settings
from repro.llm import get_client

Category = Literal["bug", "feature_request", "question", "documentation", "other"]
Source = Literal["rule_label", "rule_title", "cheap_model", "escalation_model"]

NON_BUG_LABEL = re.compile(
    r"enhancement|feature|question|documentation|\bdocs?\b|discussion|proposal|idea|support"
    r"|not[\s_-]?a[\s_-]?bug|wontfix|duplicate",
    re.IGNORECASE,
)
BUG_LABEL = re.compile(r"bug|defect|regression|crash", re.IGNORECASE)
TITLE_TAG = re.compile(
    r"^\s*[\[(]?\s*(?P<tag>bug|feature(?:[\s_-]?request)?|enhancement|question|docs?)\s*[\]):]",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You triage GitHub issues. Classify the issue into exactly one category:

- bug: existing behavior is broken or wrong. Crashes, errors, exceptions, regressions,
  incorrect output, hangs, things that "used to work", or behavior contradicting the docs.
- feature_request: asks for new behavior or a change to behavior that works as designed.
- question: asks how to do something, or is a usage/configuration problem on the user's side.
- documentation: the docs are missing, wrong or unclear (the software itself is fine).
- other: tasks, chores, discussions, release tracking, spam.

A stack trace or error message alone does not make it a bug if the user is asking for help
with their own setup. Judge by what the reporter describes, not by how they label it.
Set ambiguous to true when a reasonable maintainer could file this issue under a different
category (e.g. a docs error reported as a bug, or a feature request describing broken
behavior), and put that category in alternative.
confidence is your probability (0 to 1) that the category is correct."""


class Verdict(BaseModel):
    category: Category
    reason: str = Field(description="One short sentence")
    ambiguous: bool
    alternative: Category | None = None
    confidence: float = Field(ge=0, le=1)


class Classification(BaseModel):
    is_bug: bool
    category: Category
    confidence: float
    ambiguous: bool = False
    source: Source
    reason: str = ""
    model: str | None = None
    tokens: int = 0


def rule_classify(issue: dict[str, Any]) -> Classification | None:
    """Free tier: labels first, then title tags. None when undecided."""
    labels = issue.get("labels", [])
    non_bug = [lb for lb in labels if NON_BUG_LABEL.search(lb)]
    bug = [lb for lb in labels if BUG_LABEL.search(lb) and lb not in non_bug]
    if bug and not non_bug:
        return Classification(
            is_bug=True, category="bug", confidence=1.0, source="rule_label", reason=bug[0]
        )
    if non_bug and not bug:
        return Classification(
            is_bug=False,
            category=_category_from_tag(non_bug[0]),
            confidence=1.0,
            source="rule_label",
            reason=non_bug[0],
        )
    if not bug and (m := TITLE_TAG.match(issue.get("title", ""))):
        category = _category_from_tag(m["tag"])
        return Classification(
            is_bug=category == "bug",
            category=category,
            confidence=0.9,
            source="rule_title",
            reason=m[0].strip(),
        )
    return None


def _category_from_tag(tag: str) -> Category:
    tag = tag.lower()
    if BUG_LABEL.search(tag) and not NON_BUG_LABEL.search(tag):
        return "bug"
    if "feature" in tag or "enhancement" in tag or "proposal" in tag or "idea" in tag:
        return "feature_request"
    if "question" in tag or "support" in tag:
        return "question"
    if "doc" in tag:
        return "documentation"
    return "other"


def _content_hash(issue: dict[str, Any], use_labels: bool) -> str:
    parts = [issue.get("title", ""), issue.get("body", "")]
    if use_labels:
        parts.append(",".join(sorted(issue.get("labels", []))))
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


class BugClassifier:
    def __init__(
        self,
        *,
        provider: Provider | None = None,
        model: str | None = None,
        escalation_provider: Provider | None = None,
        escalation_model: str | None = None,
        escalate_below: float | None = None,
        use_rules: bool = True,
        use_labels: bool = True,
        max_body_chars: int = 3000,
        concurrency: int = 8,
        cache_path: str | Path | None = None,
    ) -> None:
        """use_labels=False hides labels from both the rules and the model (for evaluation)."""
        self.provider = provider or settings.classifier_provider
        self.model = model or settings.resolved_classifier_model(self.provider)
        self.escalation_provider = escalation_provider or settings.escalation_provider
        self.escalation_model = (
            escalation_model
            if escalation_model is not None
            else settings.resolved_escalation_model()
        )
        self.escalate_below = (
            escalate_below if escalate_below is not None else settings.escalate_below
        )
        self.use_rules = use_rules
        self.use_labels = use_labels
        self.max_body_chars = max_body_chars
        self.sem = asyncio.Semaphore(concurrency)
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())

    def save_cache(self) -> None:
        if self.cache_path:
            tmp = self.cache_path.with_name(self.cache_path.name + ".tmp")
            tmp.write_text(json.dumps(self.cache))
            tmp.replace(self.cache_path)

    async def classify(self, issue: dict[str, Any]) -> Classification:
        visible = issue if self.use_labels else {**issue, "labels": []}
        key = f"{issue.get('repo')}#{issue.get('number')}"
        digest = _content_hash(visible, self.use_labels)
        if (hit := self.cache.get(key)) and hit["hash"] == digest:
            return Classification.model_validate(hit["result"])

        result = rule_classify(visible) if self.use_rules else None
        if result is None:
            result = await self._ask(visible, self.provider, self.model, "cheap_model")
            if (
                self.escalation_model
                and (result.ambiguous or result.confidence < self.escalate_below)
                and self.escalation_model != self.model
            ):
                cheap_tokens = result.tokens
                result = await self._ask(
                    visible, self.escalation_provider, self.escalation_model, "escalation_model"
                )
                result.tokens += cheap_tokens

        self.cache[key] = {"hash": digest, "result": result.model_dump()}
        return result

    async def _ask(
        self, issue: dict[str, Any], provider: Provider, model: str, source: Source
    ) -> Classification:
        body = issue.get("body", "") or ""
        if len(body) > self.max_body_chars:
            body = body[: self.max_body_chars] + "\n[... truncated]"
        user = f"Title: {issue.get('title', '')}\n"
        if issue.get("labels"):
            user += f"Labels: {', '.join(issue['labels'])}\n"
        user += f"\n{body or '(no description)'}"

        async with self.sem:
            response = await get_client(provider).chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                response_format=Verdict,
            )
        verdict = response.choices[0].message.parsed
        tokens = response.usage.total_tokens if response.usage else 0
        if verdict is None:  # refusal or unparseable output: treat as unsure, not as a bug
            return Classification(
                is_bug=False,
                category="other",
                confidence=0.0,
                source=source,
                reason="no parseable verdict",
                model=model,
                tokens=tokens,
            )
        return Classification(
            is_bug=verdict.category == "bug",
            category=verdict.category,
            confidence=verdict.confidence,
            ambiguous=verdict.ambiguous,
            source=source,
            reason=verdict.reason,
            model=model,
            tokens=tokens,
        )
