# Repository rules for coding agents

These rules are mandatory for every agent working in this repository.

## Models and Git LFS

- Files under `models/` are required runtime assets, not disposable placeholders.
- A file of roughly 130 bytes beginning with
  `version https://git-lfs.github.com/spec/v1` is a Git LFS pointer. Never delete it
  as “corrupt” or “empty”.
- Before diagnosing any model problem, run:

```bash
git lfs install
git lfs pull
git lfs checkout
git lfs status
```

- Confirm the five required weights and their approximate sizes:
  ResNet50 98 MB, BLIP 945 MB, MarianMT 301 MB, EasyOCR 79 MB and 21 MB.
- Do not commit large weights as ordinary Git blobs. Keep `.gitattributes` LFS rules
  for `*.bin`, `*.pth`, `*.pt`, and `*.safetensors`.
- Do not remove a model merely because it is represented as an LFS pointer in a Git
  diff. The pointer is how LFS stores the file in Git history.

## Correct validation

- User-level Hugging Face, Torch, or EasyOCR caches are not evidence that the
  repository is deployable.
- Verify project-local files in `models/` before claiming offline deployment works.
- Run offline validation with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- A clean test must cover ResNet50, BLIP, MarianMT, EasyOCR, and importing `app`.

## Loading safety

- Keep `torch.load(..., weights_only=True)`. Do not switch it to `False` to hide a
  model error.
- If loading reports `Unsupported operand 118`, inspect the file header and run
  `git lfs pull`; PyTorch is probably reading an LFS pointer.
- Assign global model caches only after a model, processor, and tokenizer have all
  loaded successfully. Do not leave partial global state after an exception.

## Git workflow

- Inspect `git status` before pulling or editing.
- Pull with `git pull --ff-only origin main`, then run `git lfs pull`.
- Preserve unrelated user changes. Never use `git reset --hard` or delete model
  directories as a shortcut.
- Before committing, run `git diff --check`, `git lfs status`, syntax tests, and the
  offline model-loading test.

See `本地部署使用说明.md` for the user-facing installation and troubleshooting guide.
