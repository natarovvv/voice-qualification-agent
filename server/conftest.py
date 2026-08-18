"""Point every test at a throwaway data dir before config is imported."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_TMP = Path(tempfile.mkdtemp(prefix="voice-agent-test-"))
shutil.copy(HERE / "data" / "kb.json", _TMP / "kb.json")
os.environ["DATA_DIR"] = str(_TMP)
# no provider keys in tests: the offline LLM and stubs stand in
for key in ("DEEPGRAM_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
    os.environ[key] = ""
