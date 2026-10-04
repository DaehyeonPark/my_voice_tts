"""Run in the WSL project to export the approved personal voice locally."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

root = Path(__file__).resolve().parents[1]
folders = [root/'data/models/fine_tuned/dyson/active', root/'data/reference']
for folder in folders:
    if not folder.is_dir():
        raise SystemExit(f'Missing {folder}')
destination = root/'data/exports/my-voice-mac.tar'
destination.parent.mkdir(parents=True, exist_ok=True)
hashes = {}
with tempfile.TemporaryDirectory() as tmp:
    with tarfile.open(destination, 'w') as bundle:
        for folder in folders:
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    name = path.relative_to(root).as_posix()
                    with path.open('rb') as stream:
                        hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
                    bundle.add(path, arcname=name, recursive=False)
        manifest = Path(tmp)/'bundle_sha256.json'
        manifest.write_text(json.dumps(hashes, indent=2))
        bundle.add(manifest, arcname=manifest.name)
print(destination)
print(f'{destination.stat().st_size / 1024**3:.2f} GiB; keep this personal voice bundle local.')
