"""
Sphinx extension providing a directive
to render model metadata from JSON.
"""

import json

from docutils import nodes
from docutils.parsers.rst import Directive


class JsonModelTableDirective(Directive):
    """
    Render a table of models from a JSON file.

    Usage::

        .. json_models:: path/to/models.json
    """

    required_arguments = 1
    has_content = False

    def run(self):
        json_path = self.arguments[0]

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        # --- Create table structure
        table = nodes.table()
        table["classes"].append("json-model-table")

        tgroup = nodes.tgroup(cols=5)
        table += tgroup

        for _ in range(5):
            tgroup += nodes.colspec(colwidth=1)

        thead = nodes.thead()
        tbody = nodes.tbody()
        tgroup += thead
        tgroup += tbody

        # --- Header row
        header_row = nodes.row()
        for title in [
            "Model",
            r"$T_\mathrm{eff}$ range",
            r"$\lambda/\Delta\lambda$",
            "Wavelength range",
            "Reference",
        ]:
            entry = nodes.entry()
            entry += nodes.paragraph(text=title)
            header_row += entry

        thead += header_row

        # --- Data rows
        for model_key, model in data.items():
            row = nodes.row()

            # --- Model name (linked if URL exists)
            model_name = model.get("name", model_key)
            url = model.get("url")

            name_entry = nodes.entry()
            name_para = nodes.paragraph()

            if url:
                name_para += nodes.reference(
                    text=model_name,
                    refuri=url,
                    internal=False,
                )
            else:
                name_para += nodes.strong(text=model_name)

            name_entry += name_para
            row += name_entry

            # --- Teff range
            teff_entry = nodes.entry()
            teff = model.get("teff range")
            if teff:
                teff_entry += nodes.paragraph(
                    "",
                    "",
                    nodes.math(text=rf"[{teff[0]}, {teff[1]}]\,\mathrm{{K}}"),
                )
            row += teff_entry

            # --- Sampling resolution
            res_entry = nodes.entry()
            res = model.get("lambda/d_lambda")
            if res:
                res_entry += nodes.paragraph(
                    "",
                    "",
                    nodes.math(text=f"{res}"),
                )
            row += res_entry

            # --- Wavelength range
            wave_entry = nodes.entry()
            wavel_range = model.get("wavelength range")
            if wavel_range:
                wave_entry += nodes.paragraph(
                    "",
                    "",
                    nodes.math(
                        text=rf"[{wavel_range[0]}, {wavel_range[1]}]\,\mu\mathrm{{m}}"
                    ),
                )
            row += wave_entry

            # --- Reference
            ref_entry = nodes.entry()
            reference = model.get("reference")
            if reference:
                ref_entry += nodes.paragraph(text=reference)
            row += ref_entry

            tbody += row

        return [table]


def setup(app):
    """
    Register the ``json_models`` directive.
    """

    app.add_directive("json_models", JsonModelTableDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
