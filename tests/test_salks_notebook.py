import json
from pathlib import Path


def load_notebook():
    nb_path = Path(__file__).resolve().parents[1] / "Notebooks" / "Proj_2_SALKS_Chatbot.ipynb"
    assert nb_path.exists(), f"Notebook not found: {nb_path}"
    return json.loads(nb_path.read_text())


def extract_sources(nb):
    sources = []
    for cell in nb.get("cells", []):
        sources.append("".join(cell.get("source", [])))
    return "\n".join(sources)


def test_notebook_structure():
    nb = load_notebook()
    assert nb.get("nbformat") == 4
    assert len(nb.get("cells", [])) > 0


def test_notebook_contains_salks_components():
    nb = load_notebook()
    text = extract_sources(nb)

    # Core SALKS components
    assert "SMOTE" in text
    assert "KNeighborsClassifier" in text
    assert "LogisticRegression" in text
    assert "build_ann" in text

    # Chatbot functions
    assert "predict_risk" in text
    assert "chatbot" in text


def test_dataset_path_reference():
    nb = load_notebook()
    text = extract_sources(nb)
    assert "../data/heart.csv" in text
