import io
import tarfile

import pytest


@pytest.mark.parametrize("name,kind,accepted", [
    ("results/local/sample.csv", tarfile.REGTYPE, True),
    ("../escape.csv", tarfile.REGTYPE, False),
    ("results/local/link", tarfile.SYMTYPE, False),
])
def test_download_checks_archive_members(tmp_path, monkeypatch, name, kind, accepted):
    import pod
    monkeypatch.setattr(pod, "ROOT", tmp_path)
    (tmp_path / "runtime").mkdir()
    content = io.BytesIO()
    with tarfile.open(fileobj=content, mode="w:gz") as archive:
        entry = tarfile.TarInfo(name)
        entry.type = kind
        body = b"value\n1\n" if kind == tarfile.REGTYPE else b""
        entry.size = len(body)
        if kind == tarfile.SYMTYPE:
            entry.linkname = "/tmp/elsewhere"
        archive.addfile(entry, io.BytesIO(body))
    def fake_transfer(command, **kwargs):
        kwargs["stdout"].write(content.getvalue())
    monkeypatch.setattr(pod.subprocess, "run", fake_transfer)
    if accepted:
        pod.sync(dict(port=22, ip="example.invalid"), download=True)
        assert (tmp_path / name).read_text() == "value\n1\n"
    else:
        with pytest.raises(ValueError, match="Unsafe archive"):
            pod.sync(dict(port=22, ip="example.invalid"), download=True)
