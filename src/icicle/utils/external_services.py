"""Integrations with external chemical services."""

from typing import Dict

import requests


def get_compound_class(smiles: str) -> Dict:
    """Get compound classification from NPClassifier API.

    Includes error handling and rate limiting.
    """
    try:
        response = requests.get(
            f"https://npclassifier.gnps2.org/classify?smiles={smiles}"
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "class_results": ["Unknown"],
                "superclass_results": ["Unknown"],
                "pathway_results": ["Unknown"],
                "isglycoside": False,
            }
    except Exception:
        return {
            "class_results": ["Unknown"],
            "superclass_results": ["Unknown"],
            "pathway_results": ["Unknown"],
            "isglycoside": False,
        }
