"""Import the local Windows export, with safe paths and content hashes."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

root = Path(__file__).resolve().parents[1]
archive = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) == 2 else None
if archive is None or not archive.is_file():
    raise SystemExit('Usage: .venv/bin/python mac/import_voice.py /path/my-voice-mac.tar')
active = root / 'data/models/fine_tuned/dyson/active'
reference = root / 'data/reference'
if active.exists() or reference.exists():
    raise SystemExit('Existing active model/reference found. Move them aside before importing.')
with tempfile.TemporaryDirectory(dir=root) as tmp:
    stage = Path(tmp)
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts or member.issym() or member.islnk():
                raise SystemExit('Unsafe archive member')
        bundle.extractall(stage, filter='data')
    hashes = json.loads((stage/'bundle_sha256.json').read_text())
    for name, expected in hashes.items():
        target = (stage/name).resolve()
        if not target.is_relative_to(stage.resolve()):
            raise SystemExit('Unsafe manifest path')
        with target.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise SystemExit(f'Checksum mismatch: {name}')
    source = stage/'data/models/fine_tuned/dyson/active'
    for required in ['model.safetensors', 'config.json', 'model_manifest.json', 'speech_tokenizer/config.json']:
        if not (source/required).is_file():
            raise SystemExit(f'Missing model file: {required}')
    if not (stage/'data/reference/reference.json').is_file():
        raise SystemExit('Missing reference metadata')
    active.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(active))
    shutil.move(str(stage/'data/reference'), str(reference))
print('개인 목소리 설치 완료. bash mac/run.sh 를 실행하세요.')
