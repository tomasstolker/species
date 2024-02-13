"""
Module with utility functions for the ``species`` core.
"""

from typeguard import typechecked


@typechecked
def print_section(
    sect_title: str,
    bound_char: str = "-",
    extra_line: bool = True,
) -> None:
    """
    Function for printing a section title.

    Parameters
    ----------
    sect_title : str
        Section title.
    bound_char : str
        Boundary character for around the section title.
    extra_line : bool
        Extra new line at the beginning.

    Returns
    -------
    NoneType
        None
    """

    if extra_line:
        print("\n" + len(sect_title) * bound_char)
    else:
        print(len(sect_title) * bound_char)

    print(sect_title)
    print(len(sect_title) * bound_char + "\n")
