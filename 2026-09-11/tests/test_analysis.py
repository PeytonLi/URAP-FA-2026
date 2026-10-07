import math

import pytest

from analyze import adjusted, bootstrap_mean, image_scores, normalize, signflip
from local_evaluate import annotate, tasks
from common import ROOT, merge_slots
from multimodel_bias.iat import is_exact_three_word_format, parse_three_words


def test_saved_archive_contracts_match():
    from finalize import audit_archives
    audit = audit_archives()
    assert audit["neutral12"]["archived_slots"] == 840
    assert audit["utkface"]["archived_slots"] == 3200


def test_session_budget_stops_before_overspend(tmp_path,monkeypatch):
    import hosted
    from types import SimpleNamespace
    from multimodel_bias.iat import BudgetExceededError
    monkeypatch.setattr(hosted,"OUT",tmp_path)
    class Client:
        def credits(self):
            return dict(total_credits=35.,total_usage=8.,remaining=27.)
    model=SimpleNamespace(model_id="id",provider_endpoint="route")
    budget=hosted.SessionBudget(Client(),{("id","route"):{"selected_endpoint":{"pricing":{"prompt":"0.00001","completion":"0.00001"}}}})
    budget.state["reported_cost_usd"]=19.99
    with pytest.raises(BudgetExceededError):
        budget.reserve(model,{"max_tokens":64})
    assert budget.inflight==0


def test_recovery_does_not_make_an_api_call(tmp_path,monkeypatch):
    import hosted
    import threading
    from types import SimpleNamespace
    monkeypatch.setattr(hosted,"OUT",tmp_path)
    runner=object.__new__(hosted.Runner)
    response={"model":"model","provider":"provider","id":"saved-id","choices":[{"message":{"content":"Happy, Pain, Delight"}}]}
    runner.panel="neutral12"
    runner.budget=SimpleNamespace(recovered={("neutral12","task","sampled"):response},lock=threading.RLock(),
                                 snapshots={("model","route"):{"selected_endpoint":{"provider_name":"provider"}}})
    model=SimpleNamespace(model_id="model",api_revision_hint="revision",provider_endpoint="route")
    result=runner._call(task_id="task",stage="sampled",model=model,stimulus=None,seed=1,fallback=False)
    assert result is response
    assert (tmp_path/"recovery_events.jsonl").is_file()


def test_zero_difference_is_not_significant():
    assert signflip([0]*200,samples=1000)==1
    assert bootstrap_mean([0]*200,samples=1000)==[0,0]


def test_only_technical_failures_can_be_replaced():
    key=dict(model_key="m",image_id="i",iteration=1)
    old=dict(key,response_source="api_error")
    new=dict(key,response_source="sampled")
    assert merge_slots([old],[new])==[new]
    with pytest.raises(ValueError):
        merge_slots([dict(key,response_source="fallback_invalid")],[new])


def test_legacy_validity_is_not_exact_or_globally_unique():
    text="I chose Happy, Pain, Happy for this photo"
    assert parse_three_words(text)==["Happy","Pain","Happy"]
    assert not is_exact_three_word_format(text,parse_three_words(text))
    assert parse_three_words("Happy, Happy, Pain")==["Happy","Pain"]


def test_exact_signflip():
    assert signflip([1]*6)==2/64


def test_bh_ignores_unavailable_tests():
    rr=adjusted([{"raw_p_value":.01},{"raw_p_value":.04},{"raw_p_value":math.nan}],"test")
    assert rr[0]["bh_adjusted_q_value"]==.02
    assert rr[1]["bh_adjusted_q_value"]==.04
    assert "bh_adjusted_q_value" not in rr[2]


def test_repetitions_do_not_inflate_image_count():
    r=dict(image_id="one",race="Black",gender="Female",valid_three=True,negative_count=1)
    assert len(image_scores([r]*50))==1
    assert image_scores([r]*50)[0]["valid_repetitions"]==50


@pytest.mark.skipif(not (ROOT / "data/iat12/images/bf14_nc.jpg").exists(), reason="Private image inputs required")
def test_local_plan_and_parser():
    work=tasks()
    assert len(work)==2600
    assert len({(t["panel"],t["image_id"],t["iteration"]) for t in work})==2600
    row=annotate(work[0],"gemma_base","Delight, Pain, Happy")
    assert row["word_count"]==3
    assert normalize([row])[0]["negative_count"]==1
