import pytest

from multimodel_bias import openrouter


class Response:
    def __init__(self, chunks):
        self.chunks=iter(chunks)

    def __enter__(self):
        return self

    def __exit__(self,*args):
        return False

    def read1(self,size):
        return next(self.chunks,b"")


def test_json_body_can_arrive_in_chunks(monkeypatch):
    monkeypatch.setattr(openrouter.urllib.request,"urlopen",lambda *a,**k:Response([b"  ",b'{"data":',b'{"count":3}}']))
    assert openrouter.OpenRouterClient("test").get("/test")=={"data":{"count":3}}


def test_heartbeat_cannot_extend_total_read_forever(monkeypatch):
    state={"time":0}
    class Late(Response):
        def __enter__(self):
            state["time"]=121
            return self
    monkeypatch.setattr(openrouter.urllib.request,"urlopen",lambda *a,**k:Late([b" "]))
    monkeypatch.setattr(openrouter.time,"monotonic",lambda:state["time"])
    with pytest.raises(TimeoutError):
        openrouter.OpenRouterClient("test",timeout_seconds=120).get("/test")
