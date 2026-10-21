"""Offline tests for the bug classifier: rules, escalation and caching (no model calls)."""

import asyncio

from repro.bug_classifier import BugClassifier, Classification, rule_classify


def _issue(title="Something", labels=(), body="", number=1):
    return {"repo": "a/b", "number": number, "title": title, "body": body, "labels": list(labels)}


def test_rules_from_labels_and_titles():
    assert rule_classify(_issue(labels=["type/bug", "P2"])).is_bug
    assert not rule_classify(_issue(labels=["enhancement"])).is_bug
    assert rule_classify(_issue(labels=["not a bug"])).category == "other"
    assert rule_classify(_issue(title="[Bug]: crash on start")).source == "rule_title"
    assert rule_classify(_issue(title="Feature request: dark mode")).category == "feature_request"
    # Conflicting or missing signals are left to the model.
    assert rule_classify(_issue(labels=["bug", "documentation"])) is None
    assert rule_classify(_issue(title="Crash when saving", labels=["P2"])) is None


class ScriptedClassifier(BugClassifier):
    """Returns canned verdicts per model instead of calling an API."""

    def __init__(self, verdicts, **kwargs):
        super().__init__(model="cheap", escalation_model="strong", **kwargs)
        self.verdicts = verdicts
        self.calls = []

    async def _ask(self, issue, provider, model, source):
        self.calls.append(model)
        category, confidence, ambiguous = self.verdicts[model]
        return Classification(
            is_bug=category == "bug",
            category=category,
            confidence=confidence,
            ambiguous=ambiguous,
            source=source,
            model=model,
            tokens=10,
        )


def test_escalates_only_when_cheap_model_is_unsure():
    sure = ScriptedClassifier({"cheap": ("bug", 0.95, False), "strong": ("question", 0.9, False)})
    assert asyncio.run(sure.classify(_issue())).source == "cheap_model"
    assert sure.calls == ["cheap"]

    ambiguous = ScriptedClassifier(
        {"cheap": ("bug", 0.95, True), "strong": ("question", 0.9, False)}
    )
    result = asyncio.run(ambiguous.classify(_issue()))
    assert (result.source, result.is_bug, result.tokens) == ("escalation_model", False, 20)

    low = ScriptedClassifier({"cheap": ("bug", 0.4, False), "strong": ("bug", 0.9, False)})
    assert asyncio.run(low.classify(_issue())).source == "escalation_model"


def test_rules_skip_the_model_and_results_are_cached(tmp_path):
    clf = ScriptedClassifier(
        {"cheap": ("bug", 0.95, False), "strong": ("bug", 0.9, False)},
        cache_path=tmp_path / "cache.json",
    )
    assert asyncio.run(clf.classify(_issue(labels=["bug"]))).source == "rule_label"
    assert clf.calls == []

    issue = _issue(title="Crash when saving", number=2)
    asyncio.run(clf.classify(issue))
    asyncio.run(clf.classify(issue))
    assert clf.calls == ["cheap"]  # second call served from cache

    asyncio.run(clf.classify({**issue, "body": "edited"}))
    assert clf.calls == ["cheap", "cheap"]  # content changed -> reclassified

    clf.save_cache()
    reloaded = ScriptedClassifier({}, cache_path=tmp_path / "cache.json")
    assert asyncio.run(reloaded.classify({**issue, "body": "edited"})).is_bug
    assert reloaded.calls == []
