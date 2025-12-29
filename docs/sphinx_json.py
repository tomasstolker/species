"""
Sphinx extension providing a directive to render model metadata from JSON.
"""

import json

from docutils import nodes
from docutils.parsers.rst import Directive


class JsonModelListDirective(Directive):
    """
    Render a compact one-line-per-model list from a JSON file.

    Usage::

        .. json_models:: path/to/models.json
    """

    required_arguments = 1
    has_content = False

    def run(self):
        """
        Read the JSON file and generate a compact bullet list of models.
        """

        json_path = self.arguments[0]

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        bullet_list = nodes.bullet_list()

        for model_key, model in data.items():
            list_item = nodes.list_item()
            paragraph = nodes.paragraph()

            # --- Model name (linked if URL exists)
            model_name = model.get("name", model_key)
            url = model.get("url")

            if url:
                name_node = nodes.reference(
                    text=model_name,
                    refuri=url,
                    internal=False,
                )
            else:
                name_node = nodes.strong(text=model_name)

            paragraph += nodes.strong(text=model_name) if not url else name_node

            # --- Teff range
            teff = model.get("teff range")
            if teff:
                paragraph += nodes.Text(" — ")
                paragraph += nodes.math(
                    text=rf"T_{{\mathrm{{eff}}}} \in [{teff[0]}, {teff[1]}]\,\mathrm{{K}}"
                )

            # --- Reference
            reference = model.get("reference")
            if reference:
                paragraph += nodes.Text(f" — {reference}")

            list_item += paragraph
            bullet_list += list_item

        return [bullet_list]


def setup(app):
    """
    Register the ``json_models`` directive.
    """

    app.add_directive("json_models", JsonModelListDirective)

    return {
        "version": "0.2",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
